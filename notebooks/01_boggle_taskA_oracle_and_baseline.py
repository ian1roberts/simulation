# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: simulation
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task A — Oracle, Dataset, and Baseline Classifier (Keras/TensorFlow)
#
# **Goal:** Build a classical solver (trie + DFS) for a 3×3 board, generate a labelled dataset of *(board, word, path)* examples, and train a simple **baseline classifier** that predicts whether a *path exists* for a given (board, word) pair.
#
# This notebook is for Python-savvy learners who are new to deep learning. We emphasise:
# - Exact search as a ground truth **oracle**.
# - Clean data generation that **avoids leakage**.
# - A simple Keras baseline that we will **beat** in later notebooks with path-decoding models and RL.
#
# > Adjacency: we default to **8-neighbour** (like Boggle), but you can switch to 4-neighbour via a flag.
#

# %% [markdown]
# ## 0. Setup

# %%
# Python standard
import os
import random
import json
from pathlib import Path

# Numerical & plotting
import numpy as np
import matplotlib.pyplot as plt

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(tf.__version__)

# %% [markdown]
# ## 1. Dictionary loading (Ubuntu `british-english`)
# We try typical system paths and fall back to a tiny demo list if none are found.
#
# - Filters: lower-case alphabetic words of length 3–5.
# - Builds a **prefix trie** for fast prefix checks (used by the oracle and later by RL for reward shaping).
#

# %%
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
        # Fallback mini list to keep the notebook runnable anywhere.
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


WORDS = load_words()
print(f"Loaded {len(WORDS)} words (3–5 letters). Sample: {WORDS[:15]}")


# %%
# Build a simple Trie for prefix and word checks
class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children = dict()
        self.is_word = False


class Trie:
    def __init__(self, words):
        self.root = TrieNode()
        for w in words:
            self.add(w)

    def add(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def has_prefix(self, prefix):
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return False
        return True

    def is_word(self, word):
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return False
        return node.is_word


TRIE = Trie(WORDS)

# %% [markdown]
# ## 2. Board representation and oracle DFS
# - Boards are 3×3 arrays of letters: length‑9 strings or lists are fine.
# - Choose **8-neighbour** (default) or **4-neighbour** adjacency.
# - Oracle returns **(word, path)** pairs where *path* is a list of cell indices [0..8].
#

# %%
# Board utilities
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


def draw_board(board):
    # board: iterable of 9 characters
    grid = np.array(board).reshape(3, 3)
    print("\n".join(" ".join(row) for row in grid))


def plot_board(board, path=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.grid(True)
    for idx, ch in enumerate(board):
        r, c = IDX2RC[idx]
        ax.text(c, 2 - r, ch, ha="center", va="center", fontsize=18)
    if path:
        xs = [IDX2RC[i][1] for i in path]
        ys = [2 - r for r in [IDX2RC[i][0] for i in path]]
        ax.plot(xs, ys, marker="o")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    return ax


# %%
def oracle_words_on_board(board, use_diagonals=True, min_len=3, max_len=5):
    board = list(board)
    results = []  # (word, path)
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
    # De-duplicate identical words with different paths by keeping all paths (useful for training)
    return results


# Quick smoke test
board_demo = list("ratecides")  # r a t / e c i / d e s
print("Demo board:")
draw_board(board_demo)
res = oracle_words_on_board(board_demo, use_diagonals=True)
print(f"Oracle found {len(res)} (word, path) pairs; first few:", res[:5])

# %% [markdown]
# ## 3. Board sampling
# We generate random boards. For realism, you may bias letters by English frequency. Here we support both modes.
#

# %%
# Frequency from a rough approximation (can be refined); sums to 1.0
LETTER_FREQ = {
    "e": 0.127,
    "t": 0.091,
    "a": 0.082,
    "o": 0.075,
    "i": 0.070,
    "n": 0.067,
    "s": 0.063,
    "h": 0.061,
    "r": 0.060,
    "d": 0.043,
    "l": 0.040,
    "c": 0.028,
    "u": 0.028,
    "m": 0.024,
    "w": 0.024,
    "f": 0.022,
    "g": 0.020,
    "y": 0.020,
    "p": 0.019,
    "b": 0.015,
    "v": 0.010,
    "k": 0.008,
    "j": 0.002,
    "x": 0.002,
    "q": 0.001,
    "z": 0.001,
}
LETTERS = sorted(LETTER_FREQ.keys())
PROBS = np.array([LETTER_FREQ[c] for c in LETTERS], dtype=float)
PROBS = PROBS / PROBS.sum()


def sample_board(n=1, mode="frequency"):
    boards = []
    for _ in range(n):
        if mode == "uniform":
            b = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=(9,), replace=True)
        else:
            b = np.random.choice(LETTERS, size=(9,), p=PROBS, replace=True)
        boards.append(list(b))
    return boards


print("Sampled boards (first 2):")
for b in sample_board(2):
    draw_board(b)
    print()


# %% [markdown]
# ## 4. Dataset construction
# We produce a dataset of **(board, word, path)** and also create **negative pairs** (board, word with no valid path).
#
# - Train/Val/Test split at the **board level** to prevent leakage.
# - Save to `data/boggle_3x3.npz` for reuse in later notebooks.
#


# %%
def build_dataset(
    num_boards=250, mode="frequency", use_diagonals=True, min_len=3, max_len=5, neg_ratio=1.0
):
    boards = sample_board(num_boards, mode=mode)
    positive = []  # (board_str, word, path)
    neg_pairs = set()
    for b in boards:
        pairs = oracle_words_on_board(
            b, use_diagonals=use_diagonals, min_len=min_len, max_len=max_len
        )
        for w, p in pairs:
            positive.append(("".join(b), w, p))
        # Generate negatives by sampling random words of 3–5 letters with no path in this board
        if neg_ratio > 0:
            need = int(len(pairs) * neg_ratio) + 3  # ensure some negatives even if pairs is small
            tries = 0
            while need > 0 and tries < 5000:
                tries += 1
                w = random.choice(WORDS)
                # Quick check: if oracle didn't list it, it's negative
                if not any(w == wp for wp, _ in pairs):
                    neg_pairs.add(("".join(b), w))
                    need -= 1
    random.shuffle(positive)
    neg_list = list(neg_pairs)
    random.shuffle(neg_list)
    return boards, positive, neg_list


boards, positive, negatives = build_dataset(num_boards=300)
print(
    f"Dataset sizes -> boards:{len(boards)}, positives:{len(positive)}, negatives:{len(negatives)}"
)
print("Example positive:", positive[:2])
print("Example negative:", negatives[:2])


# %%
# Split by boards to avoid leakage: 70/15/15
def split_by_boards(boards, positives, negatives, train=0.7, val=0.15, test=0.15):
    board_strs = ["".join(b) for b in boards]
    random.shuffle(board_strs)
    n = len(board_strs)
    n_tr = int(n * train)
    n_va = int(n * val)
    tr_boards = set(board_strs[:n_tr])
    va_boards = set(board_strs[n_tr : n_tr + n_va])
    te_boards = set(board_strs[n_tr + n_va :])

    def filt_pos(pos_list, Bset):
        return [x for x in pos_list if x[0] in Bset]

    def filt_neg(neg_list, Bset):
        return [x for x in neg_list if x[0] in Bset]

    ds = {
        "train_pos": filt_pos(positives, tr_boards),
        "val_pos": filt_pos(positives, va_boards),
        "test_pos": filt_pos(positives, te_boards),
        "train_neg": filt_neg(negatives, tr_boards),
        "val_neg": filt_neg(negatives, va_boards),
        "test_neg": filt_neg(negatives, te_boards),
    }
    for k, v in ds.items():
        print(k, len(v))
    return ds


ds = split_by_boards(boards, positive, negatives)

# %% [markdown]
# ## 5. Vectorisation helpers
# We encode letters as integers. Words are padded to length **5**. Boards are length‑9 sequences.
#

# %%
ALPH = list("abcdefghijklmnopqrstuvwxyz")
C2I = {c: i + 1 for i, c in enumerate(ALPH)}  # 0 is PAD
I2C = {i: c for c, i in C2I.items()}
PAD = 0
MAX_WORD_LEN = 5
BOARD_LEN = 9


def encode_word(w):
    arr = [C2I.get(ch, PAD) for ch in w.lower()]
    if len(arr) < MAX_WORD_LEN:
        arr += [PAD] * (MAX_WORD_LEN - len(arr))
    return np.array(arr, dtype=np.int32)


def encode_board(bstr):
    arr = [C2I.get(ch, PAD) for ch in bstr.lower()]
    return np.array(arr, dtype=np.int32)


# %% [markdown]
# ## 6. Baseline classifier (does a path exist?)
#
# **Important:** This baseline *does not* produce a path; it only predicts if the word is findable on the board. It can “cheat” by ignoring geometry (e.g., treating the board as a bag of letters). In **Task B**, we’ll train a pointer-style model that **outputs the path** step-by-step with masking and teacher forcing.
#
# Architecture:
# - Inputs: encoded board (9 tokens) and word (padded to 5).
# - Embeddings → small Transformer encoder blocks (or BiLSTMs).
# - Concatenate pooled features → MLP → sigmoid (exist / not-exist).
#


# %%
def make_baseline_model(vocab_size=len(C2I) + 1, d_model=64):
    board_in = keras.Input(shape=(BOARD_LEN,), dtype="int32", name="board")
    word_in = keras.Input(shape=(MAX_WORD_LEN,), dtype="int32", name="word")

    emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
    b = emb(board_in)  # (None, 9, d)
    w = emb(word_in)  # (None, 5, d)

    # Simple encoders (choose LSTM for clarity)
    b_enc = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(b)
    w_enc = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(w)

    # Pool
    b_pool = layers.GlobalAveragePooling1D()(b_enc)
    w_pool = layers.GlobalAveragePooling1D()(w_enc)

    x = layers.Concatenate()([b_pool, w_pool])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[board_in, word_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


baseline = make_baseline_model()
baseline.summary()


# %% [markdown]
# ### Prepare tensors


# %%
def build_xy(pos_list, neg_list):
    boards = []
    words = []
    labels = []
    for b, w, _path in pos_list:
        boards.append(encode_board(b))
        words.append(encode_word(w))
        labels.append(1)
    for b, w in neg_list:
        boards.append(encode_board(b))
        words.append(encode_word(w))
        labels.append(0)
    Xb = np.stack(boards, axis=0)
    Xw = np.stack(words, axis=0)
    y = np.array(labels, dtype=np.float32)
    return (Xb, Xw), y


X_train, y_train = build_xy(ds["train_pos"], ds["train_neg"])
X_val, y_val = build_xy(ds["val_pos"], ds["val_neg"])
X_test, y_test = build_xy(ds["test_pos"], ds["test_neg"])

for name, (X, y) in {
    "train": (X_train, y_train),
    "val": (X_val, y_val),
    "test": (X_test, y_test),
}.items():
    print(name, X[0].shape, X[1].shape, y.shape, f"pos={y.sum()} neg={len(y) - y.sum()}")

# %% [markdown]
# ### Train baseline (small epochs to keep CPU time reasonable)

# %%
history = baseline.fit(
    {"board": X_train[0], "word": X_train[1]},
    y_train,
    validation_data=({"board": X_val[0], "word": X_val[1]}, y_val),
    epochs=5,
    batch_size=64,
    verbose=1,
)

test_metrics = baseline.evaluate({"board": X_test[0], "word": X_test[1]}, y_test, verbose=0)
print("\nTest metrics:")
for name, val in zip(baseline.metrics_names, test_metrics):
    print(f"  {name}: {val:.4f}")

# %% [markdown]
# ### Save dataset artefacts for reuse
# We store arrays and the alphabet mapping so later notebooks can load them.
#

# %%
outdir = Path("../data")
outdir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    outdir / "boggle_3x3_dataset.npz",
    Xb_train=X_train[0],
    Xw_train=X_train[1],
    y_train=y_train,
    Xb_val=X_val[0],
    Xw_val=X_val[1],
    y_val=y_val,
    Xb_test=X_test[0],
    Xw_test=X_test[1],
    y_test=y_test,
)

with open(outdir / "alphabet.json", "w") as f:
    json.dump({"alphabet": ALPH, "c2i": C2I}, f)

print(f"Saved data to {outdir.resolve()}")

# %% [markdown]
# ## What’s next (Task B)
# In **Task B**, we’ll train a **pointer-style next-move policy** that emits an explicit path (cell indices), with masking of illegal moves and teacher forcing from oracle paths. That model must respect the **geometry** of the grid, not just the spelling.
#
