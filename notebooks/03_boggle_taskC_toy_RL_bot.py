# %% [markdown]
# # Task C — Toy Reinforcement Learning: A Boggle Bot with Shaped Rewards
#
# **Goal:** Train a small policy with **REINFORCE** to explore a 3×3 board and stop when it has spelt a valid 3–5 letter word. Rewards are shaped via the dictionary **prefix trie** to mitigate sparsity.
#
# This is a pedagogical toy loop, not a performance benchmark. It illustrates:
# - State encoding,
# - Masked action spaces,
# - Reward shaping,
# - Learning curves.
#

# %% [markdown]
# ## 0. Setup

# %%
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% [markdown]
# ## 1. Dictionary, trie, and boards

# %%
# Reuse loader
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
        self.children = {}
        self.is_word = False


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

# Simple board sampler (frequency-based)
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
LETTERS = list(LETTER_FREQ.keys())
PROBS = np.array([LETTER_FREQ[c] for c in LETTERS])
PROBS = PROBS / PROBS.sum()


def sample_board():
    return "".join(np.random.choice(LETTERS, size=9, p=PROBS))


# %% [markdown]
# ## 2. Environment (masked actions, shaped rewards)
# - Actions: 0..8 select cell, 9 = STOP.
# - State: board letters, used mask, last position, current prefix.
# - Masks: cannot revisit cells; must be adjacent except on the first move.
# - Rewards:
#   - +1.0 if STOP with a valid word (length 3–5),
#   - +0.1 for staying within a valid prefix,
#   - −0.1 when prefix falls out of the trie,
#   - small −0.01 step cost to encourage shorter words.
#

# %%
IDX2RC = [(r, c) for r in range(3) for c in range(3)]
RC2IDX = {rc: i for i, rc in enumerate(IDX2RC)}


def neighbours(idx):
    r, c = IDX2RC[idx]
    nbrs = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < 3 and 0 <= cc < 3:
                nbrs.append(RC2IDX[(rr, cc)])
    return nbrs


class BoggleEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = sample_board()
        self.used = np.zeros(9, dtype=np.float32)
        self.pos = -1
        self.prefix = ""
        self.t = 0
        return self._obs()

    def _mask(self):
        legal = np.ones(10, dtype=np.float32)  # 9 cells + STOP
        for i in range(9):
            if self.used[i] == 1.0:
                legal[i] = 0.0
        if self.t > 0:
            adj = np.zeros(9, dtype=np.float32)
            for j in neighbours(self.pos):
                adj[j] = 1.0
            for i in range(9):
                legal[i] *= adj[i]
        if self.t == 0:
            legal[9] = 0.0
        return legal

    def _obs(self):
        def onehot(c):
            v = np.zeros(26, dtype=np.float32)
            v[ord(c) - 97] = 1.0
            return v

        board_vec = np.concatenate([onehot(c) for c in self.board], axis=0)
        used = self.used.astype(np.float32)
        pos = np.zeros(10, dtype=np.float32)
        pos[self.pos + 1] = 1.0  # -1 -> index 0
        pref = np.zeros(26, dtype=np.float32)
        for ch in self.prefix:
            pref[ord(ch) - 97] += 1.0
        pref = pref / (np.sum(pref) + 1e-6)
        t_vec = np.zeros(6, dtype=np.float32)
        t_vec[self.t] = 1.0
        return np.concatenate([board_vec, used, pos, pref, t_vec], axis=0), self._mask()

    def step(self, action):
        reward = -0.01  # step cost
        done = False
        info = {}
        if action == 9:  # STOP
            if 3 <= len(self.prefix) <= 5 and TRIE.is_word(self.prefix):
                reward += 1.0
            done = True
            return self._obs(), reward, done, info
        mask = self._mask()
        if mask[action] == 0.0:
            return self._obs(), -0.1, True, {"illegal": True}
        self.used[action] = 1.0
        self.pos = action
        self.prefix += self.board[action]
        self.t += 1
        if not TRIE.has_prefix(self.prefix):
            reward -= 0.1
        else:
            reward += 0.1
        if self.t >= 5:
            done = True
            if TRIE.is_word(self.prefix):
                reward += 1.0
        return self._obs(), reward, done, info


# %% [markdown]
# ## 3. Policy network and REINFORCE
# We use a small MLP that outputs 10 logits (9 cells + STOP). We apply the mask by subtracting a large constant from illegal logits.
#

# %%
OBS_DIM = len(BoggleEnv()._obs()[0])
ACT_DIM = 10


def make_policy(hidden=256):
    inp = keras.Input(shape=(OBS_DIM,), dtype="float32")
    x = layers.Dense(hidden, activation="relu")(inp)
    x = layers.Dense(hidden // 2, activation="relu")(x)
    logits = layers.Dense(ACT_DIM)(x)
    return keras.Model(inp, logits)


policy = make_policy()
optimizer = keras.optimizers.Adam(1e-3)


@tf.function
def reinforce_update(obs_batch, act_batch, ret_batch, mask_batch):
    with tf.GradientTape() as tape:
        logits = policy(obs_batch, training=True)
        masked_logits = logits + (1.0 - mask_batch) * (-1e9)
        logp = tf.nn.log_softmax(masked_logits)
        act_onehot = tf.one_hot(act_batch, depth=ACT_DIM)
        logp_act = tf.reduce_sum(act_onehot * logp, axis=1)
        loss = -tf.reduce_mean(logp_act * ret_batch)
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss


# %% [markdown]
# ## 4. Training loop & learning curve


# %%
def run_episode(env, gamma=0.99):
    obs_list = []
    act_list = []
    rew_list = []
    mask_list = []
    obs, mask = env.reset()
    done = False
    while not done:
        logits = policy(obs[None, :], training=False).numpy()[0]
        masked_logits = logits + (1.0 - mask) * (-1e9)
        probs = tf.nn.softmax(masked_logits).numpy()
        a = int(np.random.choice(np.arange(ACT_DIM), p=probs))
        obs_list.append(obs)
        act_list.append(a)
        mask_list.append(mask)
        (obs, mask), r, done, info = env.step(a)
        rew_list.append(r)
    G = 0.0
    rets = []
    for r in reversed(rew_list):
        G = r + gamma * G
        rets.append(G)
    rets = list(reversed(rets))
    return (
        np.array(obs_list),
        np.array(act_list),
        np.array(rets, dtype=np.float32),
        np.array(mask_list),
    )


env = BoggleEnv()
EPISODES = 400
mean_returns = []
success_rates = []

for ep in range(1, EPISODES + 1):
    obs, act, ret, mask = run_episode(env)
    # simple baseline: normalise returns
    ret = (ret - ret.mean()) / (ret.std() + 1e-6)
    loss = reinforce_update(
        tf.constant(obs, dtype=tf.float32),
        tf.constant(act, dtype=tf.int32),
        tf.constant(ret, dtype=tf.float32),
        tf.constant(mask, dtype=tf.float32),
    )
    mean_returns.append(float(ret.mean()))
    # success if final reward included +1 (word found); approximate check via last raw return > 0.9
    success = 1.0 if ret[-1] > 0.9 else 0.0
    success_rates.append(success)
    if ep % 50 == 0:
        print(
            f"Episode {ep}  loss={float(loss):.4f}  mean_norm_return={mean_returns[-1]:.3f}  success≈{success}"
        )

# Plot learning signals
plt.figure()
plt.plot(np.convolve(success_rates, np.ones(20) / 20, mode="valid"))
plt.title("Smoothed success indicator (~word found)")
plt.xlabel("Episode")
plt.ylabel("Success (rolling mean)")
plt.show()

# %% [markdown]
# ## Notes
# - The policy improves faster if you **initialise** it from the supervised next-move model (Task B) — a classic *behaviour cloning → RL fine-tune* pipeline.
# - Try alternative shaping, different step costs, or curriculum boards to see how exploration dynamics change.
#
