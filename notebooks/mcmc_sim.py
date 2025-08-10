# ---
# jupyter:
#   jupytext:
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
# # 🃏 Learning with Markov Chains: A Monte Carlo Simulation of Card Rank Transitions
#
# This notebook explores the foundations of **Markov Chain modeling** through the lens of card drawing — a familiar context where *randomness and dependency* intersect. Inspired by the work of **Andrey Markov**, who famously analyzed patterns in Russian literature to illustrate dependencies in sequences of letters, we apply similar reasoning to the transitions between **card ranks** in a shuffled deck.
#
# ---
#
# ## 🎯 Objective
#
# Our goal is to simulate and visualize how the rank of a card affects the rank of the card that follows it. Specifically, we:
#
# 1. **Ignore suits** and focus only on card ranks (1 to 13, Ace to King).
# 2. Simulate thousands of **shuffled deck draws** and record transitions between adjacent card ranks.
# 3. **Construct a transition matrix** to represent the empirical probabilities of one rank following another.
# 4. Use this matrix to **generate sequences** and explore how Markovian memory influences predictions.
# 5. Visualize these patterns with **heatmaps** and **distributions** to build intuitive understanding.
#
# ---
#
# ## 🧠 Why Markov Chains?
#
# A **Markov Chain** models systems where the next state depends only on the *current* state — not the full history. This is ideal for modeling card rank transitions, where our interest is in the likelihood of a particular rank appearing immediately after another, based on observed data.
#
# We use **Monte Carlo simulation** — random sampling — to empirically estimate these probabilities. Although the underlying deck is randomly shuffled, patterns emerge when we observe enough trials, allowing us to construct a **data-driven probabilistic model**.
#
# ---
#
# ## 🧪 Learning by Doing
#
# This notebook is a teaching tool — a stepwise, pedagogical journey into:
#
# - The mechanics of building a Markov model from simulation
# - The visual intuition of probability matrices
# - The power of simple systems to reveal structure through iteration
#
# It lays the groundwork for future work modeling more complex dependencies, such as the actual rules of solitaire, or stateful systems in language, genetics, or behavior.
#
# ---
#
# 🧑‍🏫 *“When we cannot clearly express a phenomenon by exact laws, we may yet hope to understand its general behavior by observing its statistical dependencies.”*
# — *after Markov, adapted for the 21st-century coder*
#

# %%
# 🧱 1. Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import collections
from tqdm import tqdm

# Define card ranks (Ace=1 to King=13)
ranks = list(range(1, 14))
n_ranks = len(ranks)
n_simulations = 10000


# %%
# Add after cell 1 - Theoretical comparison


# 🧮 1.5. Theoretical baseline comparison
def theoretical_uniform_transition():
    """Expected transition matrix for truly random card drawing"""
    return np.full((n_ranks, n_ranks), 1 / n_ranks)


theoretical_matrix = theoretical_uniform_transition()


def chi_square_test(observed, expected):
    """Simple chi-square test for goodness of fit"""
    # Check if transition_counts is defined
    if "transition_counts" not in globals():
        raise NameError("transition_counts is not defined. Please run the simulation cell first.")
    # Flatten matrices and multiply by total observations for counts
    obs_flat = observed.flatten() * transition_counts.sum()  # type: ignore
    exp_flat = expected.flatten() * transition_counts.sum()  # type: ignore

    chi2 = np.sum((obs_flat - exp_flat) ** 2 / exp_flat)
    return chi2


print("Theoretical uniform transition probability:", 1 / n_ranks)
print("This will be our baseline for comparison")
print(
    "Note: Make sure to run the simulation cell (cell 3) before using chi_square_test, so that 'transition_counts' is defined."
)

# %%
# Replace cell 2 with more efficient version

# 🎲 2. More efficient simulation with progress tracking

# Initialize transition count matrix
transition_counts = np.zeros((n_ranks, n_ranks), dtype=int)

# More efficient simulation with progress bar
print(f"Running {n_simulations} simulations...")
for sim in tqdm(range(n_simulations)):
    deck = ranks * 4
    random.shuffle(deck)

    # Vectorized transition counting
    from_ranks = np.array(deck[:-1]) - 1
    to_ranks = np.array(deck[1:]) - 1

    # Update transition counts
    for from_r, to_r in zip(from_ranks, to_ranks):
        transition_counts[from_r][to_r] += 1

print(f"Total transitions recorded: {transition_counts.sum()}")

# %%
# 📊 3. Normalize transition matrix to probabilities

transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)


# %%
# 📊 3.5. Statistical validation of transition matrix (CORRECTED)
def bootstrap_transition_matrix(n_bootstrap=100):
    """Bootstrap sampling to estimate confidence intervals"""
    bootstrap_matrices = []

    # Store all individual transitions from original simulations
    all_transitions = []
    for sim in range(n_simulations):
        deck = ranks * 4
        random.shuffle(deck)

        for i in range(len(deck) - 1):
            from_rank = deck[i] - 1
            to_rank = deck[i + 1] - 1
            all_transitions.append((from_rank, to_rank))

    # Bootstrap by resampling transitions
    for _ in range(n_bootstrap):
        # Resample transitions with replacement
        boot_transitions = np.random.choice(
            len(all_transitions), len(all_transitions), replace=True
        )

        boot_counts = np.zeros((n_ranks, n_ranks), dtype=int)
        for idx in boot_transitions:
            from_rank, to_rank = all_transitions[idx]
            boot_counts[from_rank][to_rank] += 1

        boot_probs = boot_counts / boot_counts.sum(axis=1, keepdims=True)
        bootstrap_matrices.append(boot_probs)

    return np.array(bootstrap_matrices)


# Calculate confidence intervals
bootstrap_results = bootstrap_transition_matrix()
ci_lower = np.percentile(bootstrap_results, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_results, 97.5, axis=0)

print("95% confidence intervals calculated for transition probabilities")
print(f"Bootstrap matrices shape: {bootstrap_results.shape}")

# %%
# Replace cell 4 with improved heatmap

# 🌡️ 4. Enhanced visualization with uncertainty
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Original heatmap
sns.heatmap(
    transition_probs,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    xticklabels=ranks,
    yticklabels=ranks,
    ax=ax1,
)
ax1.set_title("Empirical Transition Probabilities")
ax1.set_xlabel("Next Rank")
ax1.set_ylabel("Current Rank")

# Deviation from uniform distribution
uniform_prob = 1 / n_ranks
deviation = transition_probs - uniform_prob
sns.heatmap(
    deviation,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0,
    xticklabels=ranks,
    yticklabels=ranks,
    ax=ax2,
)
ax2.set_title("Deviation from Uniform Distribution")
ax2.set_xlabel("Next Rank")
ax2.set_ylabel("Current Rank")

plt.tight_layout()
plt.show()

# %%
# 🔮 5. Generate a single sequence of 7 cards based on transition probabilities

current_rank = random.choice(ranks)
sequence = [current_rank]

for _ in range(6):  # total of 7 cards
    prob_row = transition_probs[current_rank - 1]
    next_rank = np.random.choice(ranks, p=prob_row)
    sequence.append(next_rank)
    current_rank = next_rank

# Convert np.int64 to str for clean printout
sequence_str = [str(int(x)) for x in sequence]
print(f"Generated card rank sequence: {sequence_str}")

# %%
# 📈 6. Analyze what typically follows a specific rank, e.g. 7


def get_following_rank_distribution(start_rank, num_sequences=5000, sequence_length=7):
    followers = []

    for _ in range(num_sequences):
        current_rank = start_rank
        for _ in range(sequence_length - 1):
            prob_row = transition_probs[current_rank - 1]
            next_rank = np.random.choice(ranks, p=prob_row)
            followers.append((current_rank, next_rank))
            current_rank = next_rank

    return followers


# Example: analyze what follows a 7
start_rank = 7
followers = get_following_rank_distribution(start_rank)

# Count frequencies of each follower
following_counts = collections.Counter([b for a, b in followers if a == start_rank])
sorted_counts = dict(sorted(following_counts.items()))
df = pd.DataFrame.from_dict(sorted_counts, orient="index", columns=["Count"])

# Plot follower distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=df.index, y="Count", data=df, palette="Blues_d")
plt.title(f"Distribution of Ranks Following a {start_rank}")
plt.xlabel("Rank Following 7")
plt.ylabel("Frequency")
plt.xticks(ticks=np.arange(0, 13), labels=np.arange(1, 14))
plt.tight_layout()
plt.show()


# %%
# Add new cell for steady-state analysis


# 🔄 7. Steady-state analysis and ergodic properties
def find_steady_state(transition_matrix, max_iterations=1000, tolerance=1e-8):
    """Find the steady-state distribution of the Markov chain"""
    # Start with uniform distribution
    state = np.ones(n_ranks) / n_ranks

    for i in range(max_iterations):
        new_state = state @ transition_matrix
        if np.allclose(state, new_state, atol=tolerance):
            print(f"Steady state reached after {i+1} iterations")
            return new_state
        state = new_state

    print(f"Did not converge after {max_iterations} iterations")
    return state


steady_state = find_steady_state(transition_probs)

plt.figure(figsize=(10, 6))
plt.bar(ranks, steady_state, alpha=0.7, color="skyblue")
plt.axhline(
    y=1 / n_ranks, color="red", linestyle="--", label=f"Uniform expectation = {1/n_ranks:.3f}"
)
plt.title("Steady-State Distribution of Card Ranks")
plt.xlabel("Card Rank")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Steady-state probabilities:", np.round(steady_state, 4))

# %% [markdown]
# # ✅ Conclusion: What We Learned from the Chain
#
# In this notebook, we built a working Markov model of card rank transitions using nothing more than simulated shuffles and an empirically learned transition matrix. Along the way, we explored how:
#
# - **Markov Chains** capture dependencies between states — here, between consecutive card ranks.
# - **Monte Carlo simulations** reveal patterns through large-scale randomized sampling.
# - **Transition matrices** encode learned behavior from data, and make predictions possible.
# - **Heatmaps and distributions** provide visual intuition about how even seemingly random systems contain statistical structure.
#
# ### 📌 Key Insights
#
# - Although a shuffled deck is random, our model shows that *empirical transition probabilities* between ranks tend toward uniformity — yet small irregularities persist due to sampling variation.
# - Markov Chains give us a formal way to represent such dependencies and simulate realistic sequences without hardcoding rules.
# - Visualization (especially via Seaborn) makes abstract probabilities much more accessible and interpretable.
#
# ---
#
# ## 🔮 What's Next?
#
# This simplified exercise laid the groundwork for more complex Markovian modeling. Here are some next steps you might explore:
#
# - 🂠 **Full deck memory**: Prevent ranks from repeating in a single simulation to model a realistic deck.
# - 📜 **Rule-based modeling**: Overlay actual solitaire rules to constrain possible transitions.
# - 🧩 **Higher-order chains**: Let transitions depend on the last *two* cards instead of one.
# - 🧬 **Cross-domain learning**: Apply these techniques to other domains like genetics, weather modeling, or text prediction.
#
# ---
#
# ## 🧠 Final Thought
#
# > *"Mathematics is the art of giving the same name to different things."*
# > — Henri Poincaré
#
# What began as a game of cards became a journey through Markov’s mind. The logic you’ve practiced here — of modeling, simulating, and reflecting — applies to much more than cards. It’s a universal language for thinking clearly about systems that evolve, one step at a time.
#
# —
# *Prof. Carl & Ian, Dept. of Probabilistic Curiosity, 2025*
#
