# %% [markdown]
# # Task B — Supervised Path Decoding (Pointer-style, Keras)
#
# **Goal:** Train a model that, given a 3×3 board and a **query word (3–5 letters)**, emits a **valid path** through the grid (sequence of 0..8 indices) or declares “no path”.
#
# We implement a compact, step-wise decoder with **teacher forcing**:
# - At step *t*, score each of the 9 cells.
# - Apply masks for: used cells and non-adjacent moves from the previous pick.
# - Cross-entropy loss against the oracle path for known words.
# - Greedy decoding at inference.
#
# This forces the model to internalise **geometry + spelling**, not just dictionary membership.
#

# %% [markdown]
# ## 0. Setup & data loading

# %%
import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

CWD = Path().cwd()

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = np.load(CWD / "../data/boggle_3x3_dataset.npz")
with open(CWD / "../data/alphabet.json") as f:
    alph_info = json.load(f)
ALPH = alph_info["alphabet"]
C2I = {c: i + 1 for i, c in enumerate(ALPH)}
PAD = 0
MAX_WORD_LEN = 5
BOARD_LEN = 9

DICT_CANDIDATES = [
    "/usr/share/dict/british-english",
    "/usr/share/dict/words",
    "/usr/share/dict/american-english",
]


def load_words(paths=DICT_CANDIDATES, min_len=3, max_len=5):
    words = set()
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for w in f:
                    w = w.strip().lower()
                    if w.isalpha() and min_len <= len(w) <= max_len:
                        words.add(w)
    if not words:
        words = {
            "cat",
            "cats",
            "cater",
            "art",
            "arts",
            "rat",
            "rate",
            "rates",
            "tar",
            "tars",
            "tire",
            "tired",
            "ride",
            "rides",
            "ear",
            "ears",
            "earl",
            "ale",
            "ales",
            "tea",
            "teas",
            "eat",
            "eats",
            "seat",
            "sear",
            "scar",
            "care",
            "cared",
        }
    return sorted(words)


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children, self.is_word = {}, False


class Trie:
    def __init__(self, words):
        self.root = TrieNode()
        for w in words:
            self.add(w)

    def add(self, w):
        n = self.root
        for ch in w:
            n = n.children.setdefault(ch, TrieNode())
        n.is_word = True

    def has_prefix(self, prefix):
        n = self.root
        for ch in prefix:
            n = n.children.get(ch)
            if n is None:
                return False
        return True

    def is_word(self, w):
        n = self.root
        for ch in w:
            n = n.children.get(ch)
            if n is None:
                return False
        return n.is_word


WORDS = load_words()
TRIE = Trie(WORDS)

IDX2RC = [(r, c) for r in range(3) for c in range(3)]
RC2IDX = {rc: i for i, rc in enumerate(IDX2RC)}


def neighbours(idx, use_diagonals=True):
    r, c = IDX2RC[idx]
    nbrs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if not use_diagonals and abs(dr) + abs(dc) != 1:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < 3 and 0 <= cc < 3:
                nbrs.append(RC2IDX[(rr, cc)])
    return nbrs


def encode_word(w):
    arr = [C2I.get(ch, PAD) for ch in w.lower()]
    if len(arr) < MAX_WORD_LEN:
        arr += [PAD] * (MAX_WORD_LEN - len(arr))
    return np.array(arr, dtype=np.int32)


def encode_board(bstr):
    arr = [C2I.get(ch, PAD) for ch in bstr.lower()]
    return np.array(arr, dtype=np.int32)


# %% [markdown]
# ## 1. Rebuild training triples from oracle
# We need **(board, word, path)**. We regenerate oracle paths for the boards present in the Task‑A splits to preserve the board‑level split.
#

# %%
# Pull boards from the saved arrays
Xb_tr = data["Xb_train"]
Xb_va = data["Xb_val"]
Xb_te = data["Xb_test"]


def decode_board_tokens(tokens):
    inv = {v: k for k, v in C2I.items()}
    return "".join(inv.get(int(t), "?") for t in tokens)


train_boards = {decode_board_tokens(x) for x in Xb_tr}
val_boards = {decode_board_tokens(x) for x in Xb_va}
test_boards = {decode_board_tokens(x) for x in Xb_te}


# Oracle
def oracle_words_on_board(board, use_diagonals=True, min_len=3, max_len=5):
    board = list(board)
    results = []
    used = [False] * 9

    def dfs(idx, prefix, path):
        ch = board[idx]
        new_prefix = prefix + ch
        if not TRIE.has_prefix(new_prefix):
            return
        used[idx] = True
        path.append(idx)
        if TRIE.is_word(new_prefix) and (min_len <= len(new_prefix) <= max_len):
            results.append(("".join(new_prefix), path.copy()))
        for nb in neighbours(idx, use_diagonals=use_diagonals):
            if not used[nb]:
                dfs(nb, new_prefix, path)
        path.pop()
        used[idx] = False

    for start in range(9):
        dfs(start, "", [])
    return results


def build_triples(board_set):
    triples = []
    for b in board_set:
        pairs = oracle_words_on_board(b, use_diagonals=True)
        for w, p in pairs:
            triples.append((b, w, p))
    return triples


triples_tr = build_triples(train_boards)
triples_va = build_triples(val_boards)
triples_te = build_triples(test_boards)
print("Triples:", len(triples_tr), len(triples_va), len(triples_te))
print("Example:", triples_tr[:3])

# %% [markdown]
# ## 2. Vectorise for step-wise training
# For each triple *(b, w, path)* we create **T** training steps where *T=len(w)*. At step *t*, the target is `path[t]`.
#
# We will implement a custom training loop to apply **masks** per-example & per-step:
# - **used mask** (no repeats),
# - **adjacency mask** (must be neighbour of previous index, except at t=0 where any cell is allowed).
#

# %%
ALPH_SIZE = len(C2I) + 1
EMB = 64


def preencode(triples):
    enc = []
    for b, w, p in triples:
        enc.append((encode_board(b), encode_word(w), np.array(p, dtype=np.int32)))
    return enc


enc_tr = preencode(triples_tr)
enc_va = preencode(triples_va)
enc_te = preencode(triples_te)

ADJ = np.zeros((9, 9), dtype=np.float32)
for i in range(9):
    for j in neighbours(i, use_diagonals=True):
        ADJ[i, j] = 1.0


# %% [markdown]
# ## 3. Model definition (cell-scoring network)
#
# Given:
# - **board_emb**: (9, D)
# - **word_emb**: (5, D), we will gather char *t* and also use a pooled word embedding.
# - **prev_index**: index of previous cell (or -1 for start).
# - **step t**: optional step embedding.
#
# We compute a score for each of the 9 cells using a small MLP over concatenated features.
#


# %%
class CellScorer(keras.Model):
    def __init__(self, vocab_size, d_model=EMB):
        super().__init__()
        self.emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.pos_emb = layers.Embedding(
            input_dim=10, output_dim=d_model
        )  # 9 positions + 1 for "no prev"
        self.step_emb = layers.Embedding(input_dim=6, output_dim=d_model)  # steps 0..5
        self.w_encoder = layers.Bidirectional(layers.LSTM(d_model // 2, return_sequences=True))
        self.ff = keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),  # per-cell score
            ]
        )

    def call(self, board_tokens, word_tokens, prev_index, step_t, training=False):
        B = tf.shape(board_tokens)[0]
        b_emb = self.emb(board_tokens)  # (B,9,D)
        w_emb = self.emb(word_tokens)  # (B,5,D)
        w_seq = self.w_encoder(w_emb, training=training)  # (B,5,D)
        w_pool = tf.reduce_mean(w_seq, axis=1)  # (B,D)
        cur_idx = tf.stack([tf.range(B), step_t], axis=1)
        cur_char = tf.gather_nd(w_seq, cur_idx)  # (B,D)
        prev_index = tf.clip_by_value(prev_index, -1, 8)
        prev_pos_emb = self.pos_emb(prev_index + 1)  # shift by +1 so -1 -> 0
        step_e = self.step_emb(step_t)
        w_pool_t = tf.tile(tf.expand_dims(w_pool, axis=1), [1, 9, 1])  # (B,9,D)
        cur_char_t = tf.tile(tf.expand_dims(cur_char, axis=1), [1, 9, 1])
        prev_pos_t = tf.tile(tf.expand_dims(prev_pos_emb, axis=1), [1, 9, 1])
        step_tiled = tf.tile(tf.expand_dims(step_e, axis=1), [1, 9, 1])
        x = tf.concat([b_emb, w_pool_t, cur_char_t, prev_pos_t, step_tiled], axis=-1)  # (B,9,5D)
        scores = self.ff(x)  # (B,9,1)
        return tf.squeeze(scores, axis=-1)  # (B,9)


scorer = CellScorer(vocab_size=ALPH_SIZE)
dummy_b = tf.zeros((2, 9), dtype=tf.int32)
dummy_w = tf.zeros((2, 5), dtype=tf.int32)
_ = scorer(dummy_b, dummy_w, prev_index=tf.constant([-1, -1]), step_t=tf.constant([0, 0]))
scorer.summary()

# %% [markdown]
# ## 4. Training loop with masking
# - **used_mask**: 1 for free cells, 0 for used.
# - **adj_mask**: 1 for neighbours of prev (or all ones at t=0).
#
# Final mask = used_mask ∧ adj_mask. Illegal cells get logits `-1e9`.
#

# %%
LR = 1e-3
optimizer = keras.optimizers.Adam(LR)


@tf.function
def train_step(board_tokens, word_tokens, path_targets):
    B = tf.shape(board_tokens)[0]
    T = tf.shape(word_tokens)[1]

    total_loss = 0.0
    total_count = 0.0

    used = tf.zeros((B, 9), dtype=tf.float32)
    prev = tf.fill((B,), -1)
    for t in tf.range(T):
        tgt = path_targets[:, t]
        active = tf.where(tgt >= 0)[:, 0]
        if tf.shape(active)[0] == 0:
            break

        b_tok = tf.gather(board_tokens, active)
        w_tok = tf.gather(word_tokens, active)
        tgt_a = tf.gather(tgt, active)
        used_a = tf.gather(used, active)
        prev_a = tf.gather(prev, active)
        step_a = tf.fill((tf.shape(active)[0],), t)

        with tf.GradientTape() as tape:
            logits = scorer(b_tok, w_tok, prev_a, step_a, training=True)

            used_mask = 1.0 - used_a
            if t == 0:
                adj_mask = tf.ones_like(logits)
            else:
                adj = tf.gather(tf.constant(ADJ), prev_a)
                adj_mask = adj
            mask = used_mask * adj_mask
            masked_logits = logits + (1.0 - mask) * (-1e9)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt_a, logits=masked_logits)
            )

        grads = tape.gradient(loss, scorer.trainable_variables)
        optimizer.apply_gradients(zip(grads, scorer.trainable_variables))

        total_loss += loss * tf.cast(tf.shape(active)[0], tf.float32)
        total_count += tf.cast(tf.shape(active)[0], tf.float32)

        upd = tf.one_hot(tgt_a, depth=9, dtype=tf.float32)
        full_upd = tf.tensor_scatter_nd_update(used, tf.reshape(active, (-1, 1)), used_a + upd)
        used = full_upd
        prev = tf.tensor_scatter_nd_update(prev, tf.reshape(active, (-1, 1)), tgt_a)

    return total_loss / (total_count + 1e-6)


# %% [markdown]
# ### Build padded path tensors


# %%
def batchify(enc, batch_size=64):
    random.shuffle(enc)
    for i in range(0, len(enc), batch_size):
        batch = enc[i : i + batch_size]
        b_tok = np.stack([b for b, w, p in batch], axis=0)
        w_tok = np.stack([w for b, w, p in batch], axis=0)
        paths = []
        for b, w, p in batch:
            arr = np.full((MAX_WORD_LEN,), -1, dtype=np.int32)
            arr[: len(p)] = p
            paths.append(arr)
        y = np.stack(paths, axis=0)
        yield b_tok, w_tok, y


EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    losses = []
    for b_tok, w_tok, y in batchify(enc_tr, batch_size=64):
        loss = train_step(tf.constant(b_tok), tf.constant(w_tok), tf.constant(y))
        losses.append(float(loss.numpy()))
    print(f"Epoch {epoch}: mean loss {np.mean(losses):.4f}")


# %% [markdown]
# ## 5. Greedy decoding & evaluation


# %%
def greedy_decode(board_tokens, word_tokens):
    B = board_tokens.shape[0]
    paths = []
    for i in range(B):
        b_tok = tf.constant(board_tokens[i : i + 1])
        w_tok = tf.constant(word_tokens[i : i + 1])
        used = tf.zeros((1, 9), dtype=tf.float32)
        prev = tf.constant([-1], dtype=tf.int32)
        cur_path = []
        for t in range(MAX_WORD_LEN):
            logits = scorer(b_tok, w_tok, prev, tf.constant([t]), training=False)
            used_mask = 1.0 - used
            if t == 0:
                adj_mask = tf.ones_like(logits)
            else:
                adj_mask = tf.gather(tf.constant(ADJ), prev)
            mask = used_mask * adj_mask
            masked_logits = logits + (1.0 - mask) * (-1e9)
            idx = int(tf.argmax(masked_logits[0]).numpy())
            cur_path.append(idx)
            used = used + tf.one_hot([idx], depth=9, dtype=tf.float32)
            prev = tf.constant([idx], dtype=tf.int32)
        paths.append(cur_path)
    return paths


def evaluate_exact(triples, n_samples=500):
    sample = random.sample(triples, min(n_samples, len(triples)))
    correct = 0
    for b, w, p in sample:
        b_tok = encode_board(b)[None, :]
        w_tok = encode_word(w)[None, :]
        pred = greedy_decode(b_tok, w_tok)[0][: len(p)]
        correct += int(pred == p)
    return correct / len(sample)


acc_tr = evaluate_exact(triples_tr, n_samples=300)
acc_va = evaluate_exact(triples_va, n_samples=200)
acc_te = evaluate_exact(triples_te, n_samples=200)
print(f"Exact path match — Train:{acc_tr:.3f}  Val:{acc_va:.3f}  Test:{acc_te:.3f}")

# %% [markdown]
# ### Visualise a board and predicted path

# %%
# Small helper to print a board and overlay the predicted path
IDX2RC = [(r, c) for r in range(3) for c in range(3)]


def show_board_with_path(board_str, path):
    print("\n".join(" ".join(board_str[r * 3 : (r + 1) * 3]) for r in range(3)))
    print("Path:", path)


example = random.choice(triples_te)
b, w, p = example
print("Word:", w)
pred_path = greedy_decode(encode_board(b)[None, :], encode_word(w)[None, :])[0][: len(p)]
show_board_with_path(b, pred_path)
print("Oracle:", p, "Pred:", pred_path)

# %% [markdown]
# ## Notes
# - The model relies on **masking** to enforce legality. This is an **inductive bias**, not cheating; we still need to learn *where* to move to match the word.
# - You can try relative position features, deeper encoders, or beam search to improve exact-match accuracy.
#
