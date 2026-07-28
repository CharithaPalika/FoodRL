"""
latent_analysis.py — Probe the actor/critic bottleneck representation (z)
for hunger/satiety-like signatures, via PCA over a single rollout episode.

Requires model.py's bottleneck trunk (_BottleneckTrunk, bottleneck_dim=8 by
default) — every Actor/Critic/SharedActorCritic/SharedDiscreteActorCritic
now caches its bottleneck activation on `<module>.trunk.z` after each
forward() call. This module reads that directly — no forward hooks needed.

Usage (after you have a trained `agent`):

    from analysis.latent_analysis import capture_trunk_episode, run_pca, plot_pca_lines, plot_pca_3d

    # IMPORTANT: match deterministic= to whatever you used elsewhere
    # (e.g. agent.generate_episode() samples stochastically, deterministic=False).
    # Using a different sampling mode produces a DIFFERENT trajectory even with
    # the same env seed — the policy's RNG is separate from the env's RNG — so
    # comparing a deterministic capture against a stochastic heatmap (or vice
    # versa) will look like two unrelated episodes, not a bug in either one.
    h_seq, meta = capture_trunk_episode(agent, env, deterministic=False, network="actor")
    pcs, pca = run_pca(h_seq, n_components=3)
    plot_pca_lines(pcs, meta, agent.is_continuous)
    plot_pca_3d(pcs, meta, agent.is_continuous)

network= options:
    "actor"  (default) — agent.actor.trunk.z (non-shared) or
                          agent.policy.trunk.z (shared)
    "critic" — agent.critic.trunk.z (non-shared ONLY — shared mode has no
               separate critic trunk; requesting "critic" when shared=True
               raises, since there is nothing distinct to compare against
               "actor" in that mode — see model.py's SharedActorCritic
               docstring)

To directly compare actor vs critic latent spaces (only meaningful in
non-shared mode, since shared mode's actor/critic read off one identical z):
    h_actor,  meta = capture_trunk_episode(agent, env, network="actor")
    h_critic, _    = capture_trunk_episode(agent, env, network="critic")
    # then run_pca / plot on each, or compare directly — see
    # compare_actor_critic_bottleneck() below for a built-in side-by-side.

Shadow nutrients (fullness/hunger, etc. — see env_config.SHADOW_NUTRIENT_CONFIG):
    If the env was built with shadow nutrients configured, visualize_signature()
    automatically also plots their trend (plot_shadow_trends), reports their
    pairwise correlation (shadow_cross_correlation), and tests whether the
    chosen z/PC slice decodes them (per_shadow_decodability) — all against
    whichever network/signature/mode you already selected. This changes
    visualize_signature's return value to a 4-tuple (see its docstring):
        data, meta, pca_or_fig, shadow_results = visualize_signature(...)
    Pass analyze_shadow=False to skip this section (e.g. if env has no
    shadow nutrients, or you only want the original 3 plotted outputs).
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.utils import obs_to_tensors


# ──────────────────────────────────────────────────────────────────────────────
# Min-max normalisation helper
#
# Used throughout this module wherever a correlation or MSE comparison is
# made between two signals that live on different natural scales (e.g. a
# bottleneck dimension vs. a raw nutrient deficit, or PC1 vs. hunger). Pearson
# r is scale/shift invariant so normalisation doesn't change correlation
# values, but MSE is NOT scale invariant — comparing MSE across dims/targets
# with different native ranges is meaningless unless everything is first put
# on a common [0, 1] scale. Min-max (rather than z-score) is used because it
# gives a fixed, interpretable [0, 1] range that matches how the env itself
# normalises nutrient state (see env._load_and_normalise).
# ──────────────────────────────────────────────────────────────────────────────

def _minmax_normalize(x, axis=0, eps=1e-8):
    """
    Min-max normalise `x` to [0, 1] along `axis`.

    Parameters
    ----------
    x    : ndarray, any shape. For a (T, D) array with axis=0 (default),
           each COLUMN (e.g. each latent dimension) is normalised
           independently using its own min/max over time — this is the
           right convention for z_phy (D latent dims, each on its own
           native scale) and for a (T,) target (treated as a single column).
    axis : axis along which to take min/max (default 0 = over time/rows).
    eps  : floor on the (max - min) range to avoid divide-by-zero for a
           constant column.

    Returns
    -------
    x_norm : ndarray, same shape as x, each slice along `axis` in [0, 1]
             (a constant input column maps to all-zeros, not NaN).
    """
    x = np.asarray(x, dtype=np.float64)
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    rng = np.maximum(x_max - x_min, eps)
    return (x - x_min) / rng


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Capture z_t for every step of one episode
# ──────────────────────────────────────────────────────────────────────────────

def _get_trunk_module(agent, network="actor"):
    """
    Return the trunk module for the requested network ("actor" or
    "critic"), for whichever architecture this agent is using. Works for
    both _BottleneckTrunk (combined) and _DualBranchTrunk (separate_branches)
    — see _read_z below for how to pull the right activation out of either.
    """
    if network not in ("actor", "critic"):
        raise ValueError(f"network must be 'actor' or 'critic', got {network!r}")

    if agent.args["shared"]:
        if network == "critic":
            raise ValueError(
                "network='critic' has no meaning in shared mode — "
                "SharedActorCritic / SharedDiscreteActorCritic have ONE trunk "
                "feeding both actor_head and critic_head, so actor.trunk.z and "
                "critic.trunk.z would be identical by construction. Use a "
                "non-shared agent (PPOAgent(..., shared=False)) to get two "
                "independently-trained trunks worth comparing."
            )
        return agent.policy.trunk
    else:
        return agent.actor.trunk if network == "actor" else agent.critic.trunk


def _read_z(trunk_module, signature="combined"):
    """
    Read the requested activation off a trunk module, regardless of whether
    it's a _BottleneckTrunk (combined phy+food, single .z) or a
    _DualBranchTrunk (separate_branches=True, .z_phy / .z_food).

    signature : "combined" — the single .z (requires the trunk to be a
                              _BottleneckTrunk, i.e. separate_branches=False
                              on the network this trunk belongs to)
                "phy"      — .z_phy (requires separate_branches=True)
                "food"     — .z_food (requires separate_branches=True)
    """
    is_dual = hasattr(trunk_module, "z_phy")   # _DualBranchTrunk marker

    if signature == "combined":
        if is_dual:
            raise ValueError(
                "signature='combined' requested but this network has "
                "separate_branches=True, so there is no single combined z — "
                "use signature='phy' or signature='food' instead."
            )
        return trunk_module.z
    elif signature in ("phy", "food"):
        if not is_dual:
            raise ValueError(
                f"signature={signature!r} requested but this network has "
                "separate_branches=False, so there is only one combined z — "
                "use signature='combined' instead, or rebuild the agent with "
                "separate_branches=True if you want phy/food read separately."
            )
        return trunk_module.z_phy if signature == "phy" else trunk_module.z_food
    else:
        raise ValueError(
            f"signature must be 'combined', 'phy', or 'food', got {signature!r}"
        )


def capture_trunk_episode(agent, env, deterministic=False, max_steps=None,
                            seed=None, network="actor", signature="combined"):
    """
    Run ONE episode with the given agent/env and capture the requested
    bottleneck activation z_t at every step, read directly from the trunk.

    Parameters
    ----------
    agent        : PPOAgent (already trained / loaded, model.py bottleneck version)
    env          : FoodEnv  (same one the agent was built against)
    deterministic: bool — passed to agent.act(). True = mean/argmax policy.
    max_steps    : optional cap on episode length (defaults to env.max_steps)
    network      : "actor" (default) or "critic" — which trunk to read.
                   "critic" requires shared=False (see _get_trunk_module).
    signature    : "combined" (default) — single z from a combined phy+food
                       trunk; requires the agent's separate_branches=False.
                   "phy"  — phy-only signature; requires separate_branches=True.
                   "food" — food-only signature; requires separate_branches=True.

    Returns
    -------
    h_seq : (T, dim) float32 ndarray — z activations over time, dim is
            whichever bottleneck_dim (or phy/food_bottleneck_dim) applies.
            (kept as "h_seq" naming throughout this module for continuity
            with earlier analysis code, even though it's now a bottleneck z)
    meta  : dict of per-step arrays, all length T:
        'action'        : raw action as returned by agent.act() —
                           (num_foods,) float per step if continuous,
                           scalar int per step if discrete
        'action_sum'    : float — sum of amounts eaten this step (both modes
                           comparable: continuous = sum of fractions,
                           discrete = DISCRETE_EAT_AMOUNT or 0)
        'action_argmax' : int   — which food slot was (most) eaten, or -1 if
                           nothing was eaten this step (used for categorical
                           coloring in BOTH modes)
        'is_awake'      : bool
        'time_in_cycle' : float (normalised, from observation)
        'reward'        : float
        'phy_state'     : (num_nutrients,) float — raw physiological state
        <shadow_name>   : float per step, ONE KEY PER SHADOW NUTRIENT
                           configured in env_config.SHADOW_NUTRIENT_CONFIG
                           (e.g. 'fullness', 'hunger' if the env loaded
                           those). Only present if env.shadow_nutrient_names
                           is non-empty. These never influenced training —
                           they're read off env._shadow_nutrients purely for
                           post-hoc correlation/decodability analysis.
    """
    trunk_module = _get_trunk_module(agent, network=network)

    z_list = []
    action_list = []
    action_sum_list = []
    action_argmax_list = []
    is_awake_list = []
    time_in_cycle_list = []
    reward_list = []
    phy_list = []
    shadow_list = []   # list of dicts {shadow_name: value}, one per step
    calorie_list = []  # cumulative kcal eaten so far THIS episode (inference-only)

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    done = False
    steps = 0
    limit = max_steps if max_steps is not None else env.max_steps

    while not done and steps < limit:
        amounts_t, log_prob, value, mu_or_logits, log_std, x_raw = agent.act(
            obs, deterministic=deterministic
        )

        # agent.act() always runs a forward pass through whichever module
        # owns this trunk (actor/policy always; critic too in non-shared
        # mode, since act() calls self.critic(...) to get `value`) — so
        # the requested z is freshly populated for THIS step right now.
        z_t = _read_z(trunk_module, signature)[0].detach().cpu().numpy().copy()

        if agent.is_continuous:
            amounts_np = amounts_t.squeeze(0).detach().cpu().numpy()
            env_action = amounts_np
            action_sum = float(amounts_np.sum())
            if action_sum > 0:
                action_argmax = int(np.argmax(amounts_np))
            else:
                action_argmax = -1
            stored_action = amounts_np.copy()
        else:
            from envs.env_config import DISCRETE_EAT_AMOUNT
            action_idx = int(amounts_t.squeeze(0).detach().cpu().item())
            env_action = action_idx
            action_sum = DISCRETE_EAT_AMOUNT if action_idx > 0 else 0.0
            action_argmax = (action_idx - 1) if action_idx > 0 else -1
            stored_action = action_idx

        phy_state = np.asarray(obs["physiological_state"], dtype=np.float32)
        is_awake = bool(phy_state[-2] > 0.5)
        time_in_cycle = float(phy_state[-1])

        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        # Shadow nutrients (fullness/hunger) are computed INSIDE this same
        # env.step() call, alongside the real nutrients — so read them off
        # the env right after step() returns, which lines up with the
        # state env.step() just produced (same convention as
        # agent.generate_episode(), which reads next_obs rather than the
        # pre-step obs for its nutrient trace).
        shadow_vals = dict(getattr(env, "_shadow_nutrients", {}))

        # Cumulative calories eaten so far in THIS episode. Inference-only
        # bookkeeping (never part of observation/reward) — env resets its
        # consumption log every episode, so this rises 0 -> episode total and
        # the per-episode total is simply the last (== max) value.
        # getattr guard keeps this working with envs built before calories existed.
        _cal_fn = getattr(env, "episode_calories_consumed", None)
        calorie_list.append(float(_cal_fn()) if _cal_fn is not None else 0.0)

        z_list.append(z_t)
        action_list.append(stored_action)
        action_sum_list.append(action_sum)
        action_argmax_list.append(action_argmax)
        is_awake_list.append(is_awake)
        time_in_cycle_list.append(time_in_cycle)
        reward_list.append(float(reward))
        phy_list.append(phy_state.copy())
        shadow_list.append(shadow_vals)

        obs = next_obs
        steps += 1

    h_seq = np.stack(z_list, axis=0)  # (T, bottleneck_dim)

    meta = {
        "action": action_list,
        "action_sum": np.array(action_sum_list, dtype=np.float32),
        "action_argmax": np.array(action_argmax_list, dtype=np.int32),
        "is_awake": np.array(is_awake_list, dtype=bool),
        "time_in_cycle": np.array(time_in_cycle_list, dtype=np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "phy_state": np.stack(phy_list, axis=0),
        # Cumulative kcal eaten within the episode (see note above); per-episode
        # total = last value, per-step intake = np.diff(..., prepend=0).
        "calories_cumulative": np.array(calorie_list, dtype=np.float32),
    }

    # Shadow nutrients (fullness/hunger): one array per signal, only added
    # if the env actually has shadow nutrients configured. Each is (T,)
    # float32, aligned step-for-step with everything else in meta.
    shadow_names = list(getattr(env, "shadow_nutrient_names", []))
    for n in shadow_names:
        meta[n] = np.array([sv.get(n, 0.0) for sv in shadow_list], dtype=np.float32)

    return h_seq, meta


def capture_trunk_multi_episode(agent, env, n_episodes=20, deterministic=False,
                                  max_steps=None, network="actor", signature="combined"):
    """
    Run MULTIPLE episodes and pool the per-step (z_t, meta) data, for the
    quantitative probes in this module (which need enough samples for a
    meaningful train/test split — one ~50-200 step episode is too thin).

    network, signature : see capture_trunk_episode.

    Returns
    -------
    h_all    : (sum_T, dim) ndarray — pooled across all episodes
    meta_all : same keys as capture_trunk_episode's meta, pooled across
               episodes (concatenated along axis 0), PLUS:
        'episode_idx' : (sum_T,) int — which episode each row came from,
                        so you can group/exclude by episode if needed.
    """
    h_chunks = []
    meta_chunks = []
    ep_idx_chunks = []

    for ep in range(n_episodes):
        h_seq, meta = capture_trunk_episode(
            agent, env, deterministic=deterministic, max_steps=max_steps,
            network=network, signature=signature,
        )
        h_chunks.append(h_seq)
        meta_chunks.append(meta)
        ep_idx_chunks.append(np.full(h_seq.shape[0], ep, dtype=np.int32))

    h_all = np.concatenate(h_chunks, axis=0)

    meta_all = {}
    for key in meta_chunks[0].keys():
        if key == "action":
            # list of per-step actions (mixed shapes across modes) — just
            # concatenate the lists, not meant for numeric ops downstream.
            meta_all[key] = sum((m[key] for m in meta_chunks), [])
        else:
            meta_all[key] = np.concatenate([m[key] for m in meta_chunks], axis=0)
    meta_all["episode_idx"] = np.concatenate(ep_idx_chunks, axis=0)

    print(f"Pooled {n_episodes} episodes -> {h_all.shape[0]} total steps, "
          f"dim {h_all.shape[1]} (network={network!r}, signature={signature!r})")
    return h_all, meta_all


# ──────────────────────────────────────────────────────────────────────────────
# Actor vs Critic comparison (non-shared mode only)
# ──────────────────────────────────────────────────────────────────────────────

def compare_actor_critic_bottleneck(agent, env, n_episodes=20, deterministic=False,
                                      signature="combined"):
    """
    Capture BOTH the actor's and critic's bottleneck z, on the SAME episodes
    (same env resets, same actions taken — only the network being read
    differs), and report how similar/different their spaces are.

    Only meaningful when agent.args['shared'] is False — raises otherwise
    (see _get_trunk_module).

    signature : "combined" (default, requires agent built with
                separate_branches=False) — compares the single combined
                phy+food z between actor and critic.
                "phy" or "food" (requires separate_branches=True) — compares
                only the phy-derived or only the food-derived signature
                between actor and critic, in isolation from the other.

    What "similar" means here, concretely:
      - Per-dimension correlation between actor z and critic z is NOT
        meaningful directly (the two networks' dimensions aren't aligned —
        z[:, 0] for the actor has no reason to mean the same thing as
        z[:, 0] for the critic; they're independently initialized and
        trained). Instead we use CCA (canonical correlation analysis),
        which finds the best LINEAR ALIGNMENT between the two spaces
        and reports how correlated they are once aligned. High canonical
        correlations = the two networks converged to representing similar
        information, just in different bases. Low = they learned genuinely
        different things (actor cares about "what to do", critic cares
        about "how good is this" — these need not overlap).

    Returns
    -------
    dict with:
        'z_actor', 'z_critic'   : (T, dim) ndarrays from one combined run
                                   (env reset once per episode, both
                                   networks read on the same forward passes)
        'meta'                  : shared meta (actions taken were IDENTICAL
                                   for both, since it's one rollout)
        'canonical_correlations' : (dim,) ndarray, sorted descending — the
                                   CCA correlation for each aligned dimension
                                   pair. canonical_correlations[0] close to 1
                                   means there IS a shared direction; if ALL
                                   are low, actor and critic represent
                                   different things.
    """
    if agent.args["shared"]:
        raise ValueError(
            "compare_actor_critic_bottleneck requires shared=False — in "
            "shared mode actor and critic read off one identical z, so "
            "there's nothing to compare (see model.py SharedActorCritic)."
        )

    from sklearn.cross_decomposition import CCA

    z_actor_chunks, z_critic_chunks, meta_chunks, ep_idx_chunks = [], [], [], []

    for ep in range(n_episodes):
        # One rollout, reading BOTH trunks off the SAME forward passes —
        # agent.act() in non-shared mode always calls self.actor(...) AND
        # self.critic(...) (to get `value`), so both trunks' activations are
        # populated identically regardless of which network= we'd ask
        # capture_trunk_episode for. We inline the rollout here instead of
        # calling capture_trunk_episode twice (which would do two SEPARATE
        # episodes with different sampled actions under stochastic policies
        # — defeating the point of an apples-to-apples comparison).
        obs, _ = env.reset()
        done = False
        steps = 0
        z_a_list, z_c_list = [], []
        action_argmax_list, is_awake_list, phy_list = [], [], []

        while not done and steps < env.max_steps:
            amounts_t, log_prob, value, mu_or_logits, log_std, x_raw = agent.act(
                obs, deterministic=deterministic
            )
            z_a_list.append(_read_z(agent.actor.trunk, signature)[0].detach().cpu().numpy().copy())
            z_c_list.append(_read_z(agent.critic.trunk, signature)[0].detach().cpu().numpy().copy())

            if agent.is_continuous:
                amounts_np = amounts_t.squeeze(0).detach().cpu().numpy()
                env_action = amounts_np
                action_argmax = int(np.argmax(amounts_np)) if amounts_np.sum() > 0 else -1
            else:
                from envs.env_config import DISCRETE_EAT_AMOUNT
                action_idx = int(amounts_t.squeeze(0).detach().cpu().item())
                env_action = action_idx
                action_argmax = (action_idx - 1) if action_idx > 0 else -1

            phy_state = np.asarray(obs["physiological_state"], dtype=np.float32)
            is_awake = bool(phy_state[-2] > 0.5)

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            action_argmax_list.append(action_argmax)
            is_awake_list.append(is_awake)
            phy_list.append(phy_state.copy())
            obs = next_obs
            steps += 1

        z_actor_chunks.append(np.stack(z_a_list, axis=0))
        z_critic_chunks.append(np.stack(z_c_list, axis=0))
        meta_chunks.append({
            "action_argmax": np.array(action_argmax_list, dtype=np.int32),
            "is_awake": np.array(is_awake_list, dtype=bool),
            "phy_state": np.stack(phy_list, axis=0),
        })
        ep_idx_chunks.append(np.full(len(z_a_list), ep, dtype=np.int32))

    z_actor = np.concatenate(z_actor_chunks, axis=0)
    z_critic = np.concatenate(z_critic_chunks, axis=0)
    meta = {k: np.concatenate([m[k] for m in meta_chunks], axis=0)
            for k in meta_chunks[0].keys()}
    meta["episode_idx"] = np.concatenate(ep_idx_chunks, axis=0)

    n_components = min(z_actor.shape[1], z_critic.shape[1])
    cca = CCA(n_components=n_components)
    cca.fit(z_actor, z_critic)
    a_c, b_c = cca.transform(z_actor, z_critic)
    canonical_corrs = np.array([
        np.corrcoef(a_c[:, i], b_c[:, i])[0, 1] for i in range(n_components)
    ])
    canonical_corrs = np.sort(canonical_corrs)[::-1]

    print(f"Pooled {n_episodes} episodes -> {z_actor.shape[0]} steps.")
    print(f"Canonical correlations (actor z <-> critic z, aligned, sorted): "
          f"{np.round(canonical_corrs, 3)}")
    if canonical_corrs[0] > 0.7:
        print("  -> Strong shared direction: actor and critic converged to "
              "at least one similar representation.")
    elif canonical_corrs[0] < 0.3:
        print("  -> Weak alignment everywhere: actor and critic appear to "
              "have learned largely DIFFERENT representations.")
    else:
        print("  -> Partial alignment: some shared structure, but not a "
              "dominant shared axis.")

    return {
        "z_actor": z_actor,
        "z_critic": z_critic,
        "meta": meta,
        "canonical_correlations": canonical_corrs,
    }




def compute_nutrient_deficits(meta, env):
    """
    Compute signed deficit (target - current) for each nutrient, using the
    targets in env_config.NUTRIENT_CONFIG and the nutrient ordering in
    env.nutrient_names (this must match the column order of phy_state).

    Returns
    -------
    deficits : dict[nutrient_name] -> (T,) ndarray, signed deficit
               positive = BELOW target (deprived), negative = ABOVE target
    """
    from envs.env_config import NUTRIENT_CONFIG

    phy = meta["phy_state"]   # (T, num_nutrients + 2) — last 2 cols are is_awake, time_in_cycle
    deficits = {}
    for i, name in enumerate(env.nutrient_names):
        target = NUTRIENT_CONFIG[name]["target"]
        deficits[name] = target - phy[:, i]
    return deficits


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — PCA over the episode
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(h_seq, n_components=3):
    """
    Fit PCA on the (T, hidden_dim) trunk activations from ONE episode,
    reducing to n_components. PCA is fit across TIME within this episode —
    i.e. we're asking "what directions in the 256-dim trunk space explain
    the most variance as this single episode unfolds".

    Returns
    -------
    pcs : (T, n_components) ndarray — projected coordinates
    pca : fitted sklearn PCA object (use pca.explained_variance_ratio_ to see
          how much variance PC1/2/3 actually capture — if it's low, a 3D
          picture is not telling the whole story)
    """
    n_components = min(n_components, h_seq.shape[0], h_seq.shape[1])
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(h_seq)
    print(f"Explained variance ratio (PC1..PC{n_components}): "
          f"{np.round(pca.explained_variance_ratio_, 3)}  "
          f"(cumulative: {pca.explained_variance_ratio_.sum():.3f})")
    return pcs, pca


def run_tsne(h, n_components=2, perplexity=30.0, seed=0, standardize=True,
             max_samples=5000):
    """
    Fit t-SNE on pooled bottleneck activations (e.g. z_phy across many
    episodes), reducing to n_components (2 by default — t-SNE is almost
    always used for 2D visualisation, unlike PCA which is also useful at
    higher n_components for downstream regression).

    IMPORTANT — t-SNE is NOT a substitute for the PCA/Ridge-decodability
    pipeline elsewhere in this module:
      - It has no out-of-sample transform: there is no "fit on train,
        apply to held-out test" the way Ridge regression on raw z_phy
        supports via GroupKFold. Every point you see was used to fit the
        embedding itself.
      - It is stochastic (depends on `seed`) and famously sensitive to
        `perplexity` — distances and even relative cluster positions are
        not directly meaningful across runs or comparable in scale to PCA
        coordinates.
      - Because of the two points above, any correlation/MSE computed
        against t-SNE coordinates (see tsne_shadow_correlation_and_mse
        below) is DESCRIPTIVE ONLY: a qualitative "does this 2D layout
        visually organise itself by hunger/fullness" check, not a
        held-out claim like H1/H2 or decode_shadow_from_z_phy /
        compare_real_vs_shuffled_z_phy_decoding. Use those for anything
        you want to claim as evidence.

    IMPORTANT — COST: sklearn's TSNE is roughly O(N log N) to O(N^2)
    depending on method/backend, and on top of that holds an N x N-ish
    working set in memory during optimisation. On long episodes (e.g. a
    multi-cycle FoodEnv run with thousands of steps/episode) pooling even
    a modest number of episodes can put N in the tens or hundreds of
    thousands of rows — at which point a laptop kernel can exhaust memory
    and die silently (no Python exception, just a dead kernel process).
    To guard against this, `h` is subsampled to at most `max_samples` rows
    (deterministically, via `seed`) before t-SNE is fit. This is safe for
    a DESCRIPTIVE visualisation — t-SNE on a few thousand representative
    rows shows the same qualitative structure as fitting on everything —
    but means the embedding does not cover every row in `h`. Increase
    max_samples only if you have the memory for it (test on a notebook
    you don't mind restarting, since the failure mode is a silent kernel
    death, not a catchable error).

    Parameters
    ----------
    h            : (N, D) ndarray — pooled activations (any D), e.g. the
                   z_phy output of capture_phy_bottleneck_shadow_data /
                   capture_trunk_multi_episode.
    n_components : t-SNE output dimensionality (2 or 3 are typical; >3 is
                   unusual and rarely useful for t-SNE specifically).
    perplexity   : sklearn TSNE's perplexity parameter; capped below N/3
                   automatically so it stays valid for smaller pools.
    seed         : random_state for TSNE itself, AND the seed used for the
                   max_samples subsampling below (so the subsample chosen
                   is reproducible across runs with the same seed).
    standardize  : if True (default), each column of `h` is min-max
                   normalised (see _minmax_normalize) before fitting t-SNE,
                   so that no single latent dimension with a much larger
                   native range dominates the distance computation t-SNE
                   is built on. Set False to fit on raw h instead.
    max_samples  : hard cap on rows passed to TSNE.fit_transform. If
                   h.shape[0] exceeds this, a random subset of this size is
                   drawn (without replacement, seeded by `seed`) and a
                   warning is printed. Set to None to disable the cap and
                   fit on every row — NOT recommended for large pools; see
                   the COST note above.

    Returns
    -------
    coords     : (min(N, max_samples), n_components) ndarray — t-SNE
                 embedding coordinates for the (possibly subsampled) rows.
    tsne       : fitted sklearn TSNE object.
    sample_idx : (min(N, max_samples),) ndarray of int — row indices into
                 the ORIGINAL `h` that `coords` corresponds to. If you also
                 need to subsample `targets` for
                 tsne_shadow_correlation_and_mse/plot_tsne_scatter, index
                 with this array first (run_z_phy_shadow_analysis already
                 does this for you).
    """
    h = np.asarray(h, dtype=np.float64)
    n = h.shape[0]

    sample_idx = np.arange(n)
    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n, size=max_samples, replace=False)
        sample_idx.sort()
        print(f"[run_tsne] {n} rows exceeds max_samples={max_samples} — "
              f"subsampling to {max_samples} rows (seed={seed}) before "
              "fitting t-SNE, to avoid excessive memory/time use. Pass a "
              "larger max_samples (or None) to use more/all rows, but see "
              "this function's COST docstring note first.")
        h = h[sample_idx]
        n = h.shape[0]

    eff_perplexity = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
    if eff_perplexity != perplexity:
        print(f"[run_tsne] perplexity {perplexity} too high for N={n} rows — "
              f"using {eff_perplexity:.1f} instead.")

    h_in = _minmax_normalize(h, axis=0) if standardize else h

    tsne = TSNE(
        n_components=n_components, perplexity=eff_perplexity,
        random_state=seed, init="pca", learning_rate="auto",
    )
    coords = tsne.fit_transform(h_in)
    print(f"[run_tsne] fit on {n} rows, dim {h.shape[1]} -> {n_components} "
          f"(perplexity={eff_perplexity:.1f}, standardize={standardize}). "
          "Descriptive only — see this function's docstring before "
          "treating distances/clusters as evidence.")
    return coords, tsne, sample_idx


def tsne_shadow_correlation_and_mse(coords, targets, method="pearson"):
    """
    Descriptive-only correlation + min-max-normalised MSE between each
    t-SNE coordinate (e.g. tsne_1, tsne_2) and each shadow target
    (hunger, fullness), computed on the SAME pooled data the t-SNE
    embedding was fit on.

    This deliberately mirrors compute_z_phy_shadow_correlations' shape and
    caveats, but the caveat here is stronger: there is no held-out
    counterpart to fall back on for t-SNE specifically (no
    GroupKFold-style test is meaningful for an embedding with no
    out-of-sample transform — see run_tsne's docstring). Treat this purely
    as "does the 2D layout visually/numerically organise itself along
    hunger/fullness", not as evidence that the underlying bottleneck
    encodes them — decode_shadow_from_z_phy /
    compare_real_vs_shuffled_z_phy_decoding (run on the ORIGINAL z_phy, not
    on these t-SNE coordinates) remain the load-bearing tests for that.

    Parameters
    ----------
    coords  : (N, k) ndarray — t-SNE coordinates from run_tsne.
    targets : dict[str, (N,) ndarray] — e.g. {"hunger": ..., "fullness": ...},
              row-aligned with coords.
    method  : "pearson" or "spearman".

    Returns
    -------
    pandas.DataFrame with columns: target, tsne_dim, r, p, abs_r, mse
    (mse computed after independently min-max normalising each t-SNE
    coordinate and each target to [0, 1] — same convention as
    _per_dim_mse_table / compute_z_phy_shadow_correlations).
    """
    import pandas as pd

    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    if method == "pearson":
        from scipy.stats import pearsonr as _corr_fn
    else:
        from scipy.stats import spearmanr as _corr_fn

    coords = np.asarray(coords, dtype=np.float64)
    k = coords.shape[1]
    coords_norm = _minmax_normalize(coords, axis=0)

    rows = []
    for target_name, y in targets.items():
        y = np.asarray(y, dtype=np.float64)
        y_norm = _minmax_normalize(y, axis=0)
        for j in range(k):
            r, p = _corr_fn(coords[:, j], y)
            mse = float(np.mean((coords_norm[:, j] - y_norm) ** 2))
            rows.append({
                "target": target_name, "tsne_dim": j,
                "r": float(r), "p": float(p), "abs_r": float(abs(r)),
                "mse": mse,
            })

    df = pd.DataFrame(rows, columns=["target", "tsne_dim", "r", "p", "abs_r", "mse"])
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    print(
        "[tsne_shadow_correlation_and_mse] DESCRIPTIVE ONLY — fit and "
        "evaluated on the same data, no held-out split exists for a t-SNE "
        "embedding. Do not cite this as evidence on its own; cross-check "
        "against decode_shadow_from_z_phy / "
        "compare_real_vs_shuffled_z_phy_decoding run on the original z_phy."
    )
    print(df.round(4))
    return df


def plot_tsne_scatter(coords, targets, save_path=None):
    """
    2D scatter of t-SNE coordinates, one subplot per shadow target, points
    coloured by that target's value. Purely a visual companion to
    tsne_shadow_correlation_and_mse — if hunger/fullness organise the
    embedding, points should show a visible colour gradient across the
    layout; a salt-and-pepper pattern means they don't.

    Parameters
    ----------
    coords  : (N, 2) ndarray — t-SNE coordinates from run_tsne
              (n_components=2 required for this plot).
    targets : dict[str, (N,) ndarray] — row-aligned with coords.
    save_path : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    coords = np.asarray(coords)
    if coords.shape[1] != 2:
        raise ValueError(
            f"plot_tsne_scatter requires 2D coordinates, got shape {coords.shape}. "
            "Call run_tsne with n_components=2."
        )

    target_names = list(targets.keys())
    fig, axes = plt.subplots(1, len(target_names),
                              figsize=(5.5 * len(target_names), 5), squeeze=False)
    axes = axes[0]

    for ax, target_name in zip(axes, target_names):
        y = np.asarray(targets[target_name])
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="viridis",
                         s=10, alpha=0.7, edgecolors="none")
        fig.colorbar(sc, ax=ax, label=target_name, fraction=0.046, pad=0.04)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE of z_phy, coloured by {target_name}\n(descriptive only)",
                      fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _action_color_setup(meta, is_continuous):
    """
    Build a per-step color array + legend handles.

    discrete / continuous both use action_argmax for categorical coloring
    (-1 = ate nothing, 0..num_foods-1 = which slot was eaten), since that's
    the one coloring scheme that's meaningful in both modes.
    """
    import matplotlib.colors as mcolors

    argmax = meta["action_argmax"]
    categories = sorted(set(argmax.tolist()))
    cmap = plt.cm.tab10
    color_for = {}
    for i, cat in enumerate(categories):
        raw = "lightgray" if cat == -1 else cmap(i % 10)
        color_for[cat] = mcolors.to_rgba(raw)   # normalize to RGBA tuple

    colors = np.array([color_for[c] for c in argmax])

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_for[c],
               label=("ate nothing" if c == -1 else f"ate slot {c}"))
        for c in categories
    ]
    return colors, handles


def plot_pca_lines(pcs, meta, is_continuous, sleep_shading=True, save_path=None,
                     dim_label="PC"):
    """
    Plot each dimension of `pcs` as a line over episode steps (stacked
    subplots, one per dimension — works for any number of dimensions, not
    just 3), with markers colored by which food slot was eaten that step.

    dim_label : prefix used for y-axis labels and the title — "PC" (default,
                use when `pcs` came from run_pca) or "dim" / anything else
                when plotting RAW dimensions that were never PCA'd (passing
                "PC" in that case would misleadingly imply PCA was applied).
    """
    T = pcs.shape[0]
    n_dims = pcs.shape[1]
    t = np.arange(T)
    colors, handles = _action_color_setup(meta, is_continuous)

    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(t, pcs[:, i], color="steelblue", lw=1.0, alpha=0.6, zorder=1)
        ax.scatter(t, pcs[:, i], c=colors, s=18, zorder=2, edgecolors="none")
        ax.set_ylabel(f"{dim_label}{i+1}")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        if sleep_shading:
            awake = meta["is_awake"]
            in_sleep = False
            start = None
            for step in range(T):
                if not awake[step] and not in_sleep:
                    in_sleep, start = True, step
                elif awake[step] and in_sleep:
                    ax.axvspan(start, step, color="gray", alpha=0.08, zorder=0)
                    in_sleep = False
            if in_sleep:
                ax.axvspan(start, T, color="gray", alpha=0.08, zorder=0)

    axes[-1].set_xlabel("Episode step")
    title_prefix = "PCA components" if dim_label == "PC" else "Raw dimensions"
    axes[0].set_title(f"Actor trunk hidden state — {title_prefix} over one episode\n"
                       "(gray bands = asleep, markers colored by food slot eaten)")
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_pca_3d(pcs, meta, is_continuous, save_path=None, dim_label="PC"):
    """
    Plot the 3D trajectory through the given 3-dimensional space over the
    episode, colored by which food slot was eaten that step. The line shows
    the PATH; markers highlight eating events.

    dim_label : "PC" (default, for PCA output) or "dim" (for raw,
                non-PCA'd dimensions) — controls axis labels and title.
    """
    colors, handles = _action_color_setup(meta, is_continuous)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory line, colored by time (so direction of travel is visible)
    T = pcs.shape[0]
    for i in range(T - 1):
        ax.plot(pcs[i:i+2, 0], pcs[i:i+2, 1], pcs[i:i+2, 2],
                 color=plt.cm.viridis(i / max(T - 1, 1)), lw=1.2, alpha=0.7)

    # Eating-event markers on top
    ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=colors, s=25,
               edgecolors="none", depthshade=True)

    ax.set_xlabel(f"{dim_label}1", labelpad=10)
    ax.set_ylabel(f"{dim_label}2", labelpad=10)
    ax.set_zlabel(f"{dim_label}3", labelpad=10)
    title_prefix = "PCA-space" if dim_label == "PC" else "Raw-dimension"
    ax.set_title(f"Actor trunk {title_prefix} trajectory (one episode)\n"
                 "line color = time progression (dark→light = early→late), "
                 "markers = food slot eaten")
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_nutrients_and_pcs(pcs, meta, env, is_continuous, n_pcs=3, sleep_shading=True, save_path=None):
    """
    Stacked plot: each nutrient's raw level (glucose/peptides/fatty_acids —
    i.e. carbs/protein/fats proxies) on top, then PC1..PC{n_pcs} below,
    all sharing the x-axis (episode step) so you can visually check whether
    PC movements track nutrient swings. Eating events marked on every panel.
    """
    nutrient_names = env.nutrient_names
    n_nutrients = len(nutrient_names)
    colors, handles = _action_color_setup(meta, is_continuous)
    T = pcs.shape[0]
    t = np.arange(T)

    n_panels = n_nutrients + n_pcs
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.2 * n_panels), sharex=True)

    def _shade_sleep(ax):
        if not sleep_shading:
            return
        awake = meta["is_awake"]
        in_sleep, start = False, None
        for step in range(T):
            if not awake[step] and not in_sleep:
                in_sleep, start = True, step
            elif awake[step] and in_sleep:
                ax.axvspan(start, step, color="gray", alpha=0.08, zorder=0)
                in_sleep = False
        if in_sleep:
            ax.axvspan(start, T, color="gray", alpha=0.08, zorder=0)

    # ── Nutrient panels ────────────────────────────────────────────────────
    for i, name in enumerate(nutrient_names):
        ax = axes[i]
        ax.plot(t, meta["phy_state"][:, i], color="darkgreen", lw=1.2, alpha=0.7, zorder=1)
        ax.scatter(t, meta["phy_state"][:, i], c=colors, s=16, zorder=2, edgecolors="none")
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        _shade_sleep(ax)

    # ── PC panels ──────────────────────────────────────────────────────────
    for j in range(n_pcs):
        ax = axes[n_nutrients + j]
        ax.plot(t, pcs[:, j], color="steelblue", lw=1.0, alpha=0.6, zorder=1)
        ax.scatter(t, pcs[:, j], c=colors, s=16, zorder=2, edgecolors="none")
        ax.set_ylabel(f"PC{j+1}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        _shade_sleep(ax)

    axes[-1].set_xlabel("Episode step")
    axes[0].set_title(
        "Nutrient levels (carbs≈glucose, protein≈peptides, fats≈fatty_acids) "
        f"vs. trunk PC1..PC{n_pcs}\n(gray = asleep, markers = food slot eaten)"
    )
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_shadow_trends(meta, is_continuous, shadow_names=None, sleep_shading=True,
                         save_path=None):
    """
    Plot shadow nutrient trends (fullness / hunger, or whichever keys
    env_config.SHADOW_NUTRIENT_CONFIG defines) over one episode — one
    stacked line+scatter panel per shadow signal, same visual convention as
    plot_pca_lines / plot_nutrients_and_pcs: a line for the trend, markers
    on top colored by which food slot was eaten that step (eating points
    as scatter), sleep windows shaded gray.

    This is purely descriptive — it does not run PCA or any correlation
    test; see per_shadow_decodability() for the quantitative side.

    Parameters
    ----------
    meta          : dict from capture_trunk_episode / capture_trunk_multi_episode.
                    Must contain the requested shadow_names as keys (added
                    automatically by capture_trunk_episode when the env has
                    shadow_nutrient_names configured).
    shadow_names  : list of meta keys to plot (default: every key in meta
                    that isn't one of the known non-shadow keys — i.e.
                    auto-detects 'fullness'/'hunger'/etc.). Pass explicitly
                    if you only want a subset.
    """
    known_keys = {
        "action", "action_sum", "action_argmax", "is_awake",
        "time_in_cycle", "reward", "phy_state", "episode_idx",
    }
    if shadow_names is None:
        shadow_names = [k for k in meta.keys() if k not in known_keys]
    if not shadow_names:
        raise ValueError(
            "No shadow nutrient keys found in meta — did capture_trunk_episode "
            "run against an env with shadow_nutrient_names configured "
            "(env_config.SHADOW_NUTRIENT_CONFIG)?"
        )
    missing = [n for n in shadow_names if n not in meta]
    if missing:
        raise ValueError(f"shadow_names {missing} not present in meta.")

    colors, handles = _action_color_setup(meta, is_continuous)
    T = len(meta[shadow_names[0]])
    t = np.arange(T)

    fig, axes = plt.subplots(len(shadow_names), 1,
                               figsize=(14, 3 * len(shadow_names)), sharex=True)
    axes = np.atleast_1d(axes)

    def _shade_sleep(ax):
        if not sleep_shading:
            return
        awake = meta["is_awake"]
        in_sleep, start = False, None
        for step in range(T):
            if not awake[step] and not in_sleep:
                in_sleep, start = True, step
            elif awake[step] and in_sleep:
                ax.axvspan(start, step, color="gray", alpha=0.08, zorder=0)
                in_sleep = False
        if in_sleep:
            ax.axvspan(start, T, color="gray", alpha=0.08, zorder=0)

    for i, name in enumerate(shadow_names):
        ax = axes[i]
        ax.plot(t, meta[name], color="darkorange", lw=1.2, alpha=0.7, zorder=1)
        ax.scatter(t, meta[name], c=colors, s=18, zorder=2, edgecolors="none")
        ax.set_ylabel(name, fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        _shade_sleep(ax)

    axes[-1].set_xlabel("Episode step")
    axes[0].set_title(
        f"Shadow nutrient trend(s): {', '.join(shadow_names)}\n"
        "(gray = asleep, markers = food slot eaten — not used in training "
        "or reward)"
    )
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def shadow_cross_correlation(meta, shadow_names=None):
    """
    Pearson correlation between every pair of shadow nutrients over one
    episode (e.g. fullness vs hunger) — a sanity check before reading
    anything into the bottleneck comparison: if fullness/hunger don't even
    correlate sensibly with EACH OTHER (expected: strongly negative — full
    means not hungry), the reconstructed shadow trace itself is suspect,
    independent of whatever z does or doesn't encode.

    Returns
    -------
    pandas.DataFrame, square matrix of Pearson r between each pair.
    """
    import pandas as pd
    from scipy.stats import pearsonr

    known_keys = {
        "action", "action_sum", "action_argmax", "is_awake",
        "time_in_cycle", "reward", "phy_state", "episode_idx",
    }
    if shadow_names is None:
        shadow_names = [k for k in meta.keys() if k not in known_keys]

    n = len(shadow_names)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, p = pearsonr(meta[shadow_names[i]], meta[shadow_names[j]])
            mat[i, j] = mat[j, i] = r
            flag = "*" if p < 0.05 else ""
            print(f"Pearson r({shadow_names[i]}, {shadow_names[j]}) = {r:+.3f}{flag}  (p={p:.2e})")

    return pd.DataFrame(mat, index=shadow_names, columns=shadow_names)


def plot_pca_grid(h_seq, meta, is_continuous, n_components=10, save_path=None):
    """
    Fit PCA with n_components (default 10) and plot each PC as a small
    subplot in a grid, colored by food slot eaten. Useful for scanning
    beyond PC1-3 for a candidate "hormonal" direction, since PCA orders by
    VARIANCE, not by relevance — a meaningful signal could live in PC7.

    Returns
    -------
    pcs : (T, n_components) ndarray
    pca : fitted sklearn PCA object
    """
    n_components = min(n_components, h_seq.shape[0], h_seq.shape[1])
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(h_seq)
    print(f"Explained variance ratio (PC1..PC{n_components}): "
          f"{np.round(pca.explained_variance_ratio_, 3)}  "
          f"(cumulative: {pca.explained_variance_ratio_.sum():.3f})")

    colors, handles = _action_color_setup(meta, is_continuous)
    T = pcs.shape[0]
    t = np.arange(T)

    ncols = 2
    nrows = int(np.ceil(n_components / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.0 * nrows), sharex=True)
    axes = axes.flatten()

    for i in range(n_components):
        ax = axes[i]
        ax.plot(t, pcs[:, i], color="steelblue", lw=0.9, alpha=0.6, zorder=1)
        ax.scatter(t, pcs[:, i], c=colors, s=10, zorder=2, edgecolors="none")
        ax.set_ylabel(f"PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.1f}%)", fontsize=8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for i in range(n_components, len(axes)):
        axes[i].axis("off")

    axes[0].legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(f"Top {n_components} trunk PCs over one episode "
                 "(% = variance explained by that PC)", y=1.0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return pcs, pca



    """
    Plot the (PC1, PC2, PC3) trajectory through 3D space over the episode,
    colored by which food slot was eaten that step. The line shows the
    PATH; markers highlight eating events.
    """
    colors, handles = _action_color_setup(meta, is_continuous)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory line, colored by time (so direction of travel is visible)
    T = pcs.shape[0]
    for i in range(T - 1):
        ax.plot(pcs[i:i+2, 0], pcs[i:i+2, 1], pcs[i:i+2, 2],
                 color=plt.cm.viridis(i / max(T - 1, 1)), lw=1.2, alpha=0.7)

    # Eating-event markers on top
    ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], c=colors, s=25,
               edgecolors="none", depthshade=True)

    ax.set_xlabel("PC1", labelpad=10)
    ax.set_ylabel("PC2", labelpad=10)
    ax.set_zlabel("PC3", labelpad=10)
    ax.set_title("Actor trunk hidden-state trajectory (one episode)\n"
                 "line color = time progression (dark→light = early→late), "
                 "markers = food slot eaten")
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Quantitative "hormonal signature" test suite
#
# Visual PCA is suggestive, not evidence. This section tests two concrete,
# falsifiable hypotheses using the POOLED multi-episode data from
# capture_trunk_multi_episode (single-episode data is too thin for a
# meaningful train/test split):
#
#   H1 (tracks deficit):
#       Some direction in h correlates with / linearly predicts nutrient
#       deficit (target - current_level). Tested via:
#         (a) per-PC Pearson correlation against each nutrient's deficit
#             (cheap, but PCA's top components are chosen by VARIANCE, not
#             relevance to deficit — a real signal could live outside PC1-3)
#         (b) ridge regression of the FULL h (all hidden_dim dims, not just
#             top PCs) onto each nutrient's deficit, with train/test split,
#             reporting held-out R². This is the rigorous version of H1.
#
#   H2 (predicts action, beyond the raw observation):
#       h contains information about the NEXT action that isn't already in
#       phy_state (the raw observation) alone. Tested by comparing two
#       probes' held-out performance:
#         - probe A: phy_state[t]  -> action_sum[t+1]  (or action_argmax[t+1])
#         - probe B: h[t]          -> action_sum[t+1]  (or action_argmax[t+1])
#       If probe B clearly beats probe A out-of-sample, the trunk built
#       something predictive beyond a pass-through of the input — that's the
#       closest thing here to genuine evidence of an emergent "anticipatory"
#       signal rather than a relabeling of what was already in phy_state.
#
# Both use train/test splits BY EPISODE (not by random row), since rows
# within an episode are highly autocorrelated — a random row split would
# leak information across train/test and inflate R² misleadingly.
# ══════════════════════════════════════════════════════════════════════════════

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score
from scipy.stats import pearsonr


def _per_dim_mse_table(h, targets, index_labels=None, label_fmt="PC{j}"):
    """
    Shared helper: mean-squared error between EVERY column of `h` (e.g.
    every PC or every latent dim) and EVERY target signal, both put on a
    common [0, 1] scale first via min-max normalisation (see
    _minmax_normalize). MSE is not scale invariant the way Pearson r is, so
    this normalisation step is required for the numbers to be comparable
    across dims/targets that live on different native scales.

    This mirrors the shape/orientation of the correlation tables in this
    module (rows = targets, cols = dim labels) so the two can be read
    side by side or plotted with the same heatmap layout.

    Parameters
    ----------
    h            : (T, D) ndarray — raw (NOT pre-normalised) latent/PC array.
    targets      : dict[name] -> (T,) ndarray — raw (NOT pre-normalised)
                   target signals, e.g. nutrient deficits or shadow levels.
    index_labels : optional list of D column labels; defaults to
                   [label_fmt.format(j=j+1) for j in range(D)].
    label_fmt    : format string for default column labels (1-indexed `j`).

    Returns
    -------
    pandas.DataFrame, rows = nutrients/targets, cols = dim labels,
    values = MSE(normalised h[:, j], normalised target) — lower is better
    (0 = perfect match after normalisation, NOT the same thing as "high
    correlation"; a column and a target can be perfectly correlated but
    still have nonzero MSE if their normalised curves are offset/scaled
    differently within [0, 1] — see this function's caller docstrings for
    how to read MSE alongside r rather than instead of it).
    """
    import pandas as pd

    h = np.asarray(h, dtype=np.float64)
    n_dims = h.shape[1]
    h_norm = _minmax_normalize(h, axis=0)   # each column independently -> [0,1]

    if index_labels is None:
        index_labels = [label_fmt.format(j=j + 1) for j in range(n_dims)]

    rows = {}
    for name, y in targets.items():
        y = np.asarray(y, dtype=np.float64)
        y_norm = _minmax_normalize(y, axis=0)   # (T,) -> (T,) in [0,1]
        row = [
            float(np.mean((h_norm[:, j] - y_norm) ** 2))
            for j in range(n_dims)
        ]
        rows[name] = row

    df = pd.DataFrame(rows, index=index_labels).T
    return df


def _plot_mse_heatmap(mse_df, title, save_path=None):
    """
    Heatmap of a _per_dim_mse_table-shaped DataFrame (rows = targets,
    cols = dim labels). Lower MSE = better match after min-max
    normalisation, so this uses a sequential colormap (darker = higher
    error), unlike the diverging RdBu_r used for correlation (where sign
    matters). Shares layout conventions with
    plot_z_phy_shadow_correlation_heatmap so the two can be compared side
    by side.
    """
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * mse_df.shape[1]), max(3, 0.6 * mse_df.shape[0] + 1.5))
    )
    vmax = float(np.nanmax(mse_df.values)) if mse_df.size else 1.0
    vmax = max(vmax, 1e-6)

    im = ax.imshow(mse_df.values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(mse_df.shape[1]))
    ax.set_xticklabels(mse_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(mse_df.shape[0]))
    ax.set_yticklabels(mse_df.index, fontsize=10)
    ax.set_title(title, fontsize=12)

    for i in range(mse_df.shape[0]):
        for j in range(mse_df.shape[1]):
            val = mse_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if val > 0.6 * vmax else "black")

    fig.colorbar(im, ax=ax, label="MSE (min-max normalised)", fraction=0.04, pad=0.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def per_pc_deficit_correlation(pcs, deficits):
    """
    H1(a) — cheap screen: Pearson correlation of each PC against each
    nutrient's deficit, AND mean-squared error between the two after
    min-max normalising both to [0, 1] (correlation is scale-invariant so
    it's computed on the raw PCs/deficits as before; MSE is NOT
    scale-invariant, so it needs the normalisation step to be meaningful
    across PCs/nutrients with different native ranges — see
    _per_dim_mse_table).

    Use this to decide which PCs/nutrients are worth a closer look, NOT as
    a final answer (PCA's top components aren't chosen for relevance to
    deficit, and a low MSE alongside a near-zero correlation usually just
    means both signals happen to occupy similar ranges, not that they
    track each other).

    Parameters
    ----------
    pcs      : (T, n_components) ndarray — from run_pca / plot_pca_grid
    deficits : dict[name] -> (T,) ndarray — from compute_nutrient_deficits

    Returns
    -------
    (corr_df, mse_df) : both pandas.DataFrame, rows = nutrients,
    cols = PC1..PCk.
        corr_df : values = Pearson r (also prints p-values flagged where
                  r is significant at p<0.05).
        mse_df  : values = MSE(min-max normalised PC, min-max normalised
                  deficit) — lower is better, not directly comparable in
                  magnitude to r.
    """
    import pandas as pd

    n_components = pcs.shape[1]
    rows = {}
    for name, defc in deficits.items():
        row = []
        for j in range(n_components):
            r, p = pearsonr(pcs[:, j], defc)
            flag = "*" if p < 0.05 else ""
            row.append(f"{r:+.3f}{flag}")
        rows[name] = row

    df = pd.DataFrame(rows, index=[f"PC{j+1}" for j in range(n_components)]).T
    print("Pearson r of each nutrient's deficit vs each PC ('*' = p<0.05):")
    print(df)

    mse_df = _per_dim_mse_table(pcs, deficits, label_fmt="PC{j}")
    print("\nMSE of each nutrient's deficit vs each PC (min-max normalised, lower=better):")
    print(mse_df.round(4))

    return df, mse_df


def _episode_group_kfold_r2(X, y, groups, n_splits=5, alpha=1.0):
    """
    Ridge regression with GroupKFold (grouped by episode_idx) to get an
    honest held-out R². Returns mean and per-fold R² so you can see variance
    across folds, not just a single optimistic number.
    """
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 episodes (got {n_groups}) for a grouped "
            "train/test split — increase n_episodes in capture_trunk_multi_episode."
        )

    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model = Ridge(alpha=alpha)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_r2.append(r2_score(y[test_idx], pred))
    return float(np.mean(fold_r2)), fold_r2


def test_h1_deficit_decodability(h_all, meta_all, env, alpha=1.0, n_splits=5):
    """
    H1, rigorous version — ridge regression of FULL h onto each nutrient's
    deficit, held-out R² via GroupKFold (grouped by episode so train/test
    don't share autocorrelated rows from the same episode).

    A held-out R² well above 0 means deficit is linearly decodable from h.
    R² near 0 or negative means it isn't (at least not linearly) — a real
    finding either way, not a failure of the test.

    Returns
    -------
    pandas.DataFrame: nutrient -> mean held-out R², per-fold R² list
    """
    import pandas as pd

    deficits = compute_nutrient_deficits(meta_all, env)
    groups = meta_all["episode_idx"]

    results = {}
    for name, defc in deficits.items():
        mean_r2, fold_r2 = _episode_group_kfold_r2(h_all, defc, groups,
                                                      n_splits=n_splits, alpha=alpha)
        results[name] = {"mean_r2": mean_r2, "fold_r2": np.round(fold_r2, 3)}
        print(f"H1 [{name:12s}] held-out R² = {mean_r2:+.3f}  "
              f"(folds: {np.round(fold_r2, 3)})")

    return pd.DataFrame(results).T


def per_shadow_decodability(h_all, meta_all, shadow_names=None, alpha=1.0, n_splits=5):
    """
    H1-style test, but for shadow nutrients instead of a regulated
    nutrient's deficit: ridge regression of the FULL probed representation
    (h_all — whatever z/PC slice the caller already chose) onto the
    ABSOLUTE LEVEL of each shadow nutrient, held-out R² via GroupKFold
    grouped by episode (same machinery as test_h1_deficit_decodability —
    see _episode_group_kfold_r2).

    IMPORTANT — this is a weaker claim than test_h1_deficit_decodability:
    glucose/peptides/fatty_acids have a defined target, so H1 asks "is the
    DEFICIT FROM TARGET decodable". Fullness/hunger have no target (they
    were never part of NUTRIENT_CONFIG, never regulated, never rewarded) —
    so this can only ask "is the RAW LEVEL decodable". A positive result
    here is evidence the representation correlates with fullness/hunger;
    it is not directly comparable to the H1 numbers without keeping that
    distinction in mind.

    Requires h_all/meta_all from capture_trunk_multi_episode (multiple
    episodes pooled) — a single episode doesn't have enough independent
    groups for GroupKFold.

    Returns
    -------
    pandas.DataFrame: shadow_name -> mean held-out R², per-fold R² list
    """
    import pandas as pd

    known_keys = {
        "action", "action_sum", "action_argmax", "is_awake",
        "time_in_cycle", "reward", "phy_state", "episode_idx",
    }
    if shadow_names is None:
        shadow_names = [k for k in meta_all.keys() if k not in known_keys]
    if not shadow_names:
        raise ValueError(
            "No shadow nutrient keys found in meta_all — did "
            "capture_trunk_multi_episode run against an env with "
            "shadow_nutrient_names configured?"
        )

    groups = meta_all["episode_idx"]
    results = {}
    for name in shadow_names:
        y = meta_all[name]
        mean_r2, fold_r2 = _episode_group_kfold_r2(h_all, y, groups,
                                                      n_splits=n_splits, alpha=alpha)
        results[name] = {"mean_r2": mean_r2, "fold_r2": np.round(fold_r2, 3)}
        print(f"Shadow-decodability [{name:10s}] held-out R² = {mean_r2:+.3f}  "
              f"(folds: {np.round(fold_r2, 3)})  "
              "[raw level, no target — not directly comparable to H1's deficit R²]")

    return pd.DataFrame(results).T


def test_h2_action_predictivity(h_all, meta_all, alpha=1.0, n_splits=5,
                                  target="action_sum"):
    """
    H2 — does h[t] predict the NEXT step's action better than phy_state[t]
    alone does? Compares two probes with the SAME train/test split, so the
    comparison is apples to apples.

    Parameters
    ----------
    target : 'action_sum' (regression, continuous amount) or
              'action_argmax' (classification, which slot — treated as
              categorical; -1 'ate nothing' is its own class)

    Returns
    -------
    dict with 'phy_state_score' and 'h_score' (mean held-out R² or balanced
    accuracy depending on target), the per-fold breakdown for both, and for
    classification also 'majority_baseline' (balanced accuracy of always
    predicting the most common class — if phy_state_score/h_score are at or
    near this floor, NEITHER probe learned anything; that's a different
    conclusion than "h has no advantage", and worth checking explicitly
    rather than reading the difference alone).
    """
    groups_full = meta_all["episode_idx"]
    phy = meta_all["phy_state"]

    # Shift: predict action at t+1 from state at t. Drop the last step of
    # EACH episode (no t+1 available) rather than naively shifting the
    # pooled array, which would wrongly pair the last step of episode k
    # with the first step of episode k+1.
    X_h_list, X_phy_list, y_list, g_list = [], [], [], []
    for ep in np.unique(groups_full):
        mask = groups_full == ep
        h_ep = h_all[mask]
        phy_ep = phy[mask]
        y_ep = meta_all[target][mask]

        if len(y_ep) < 2:
            continue
        X_h_list.append(h_ep[:-1])
        X_phy_list.append(phy_ep[:-1])
        y_list.append(y_ep[1:])
        g_list.append(np.full(len(y_ep) - 1, ep))

    X_h = np.concatenate(X_h_list, axis=0)
    X_phy = np.concatenate(X_phy_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    n_groups = len(np.unique(groups))
    n_splits_eff = min(n_splits, n_groups)
    if n_splits_eff < 2:
        raise ValueError(
            f"Need at least 2 episodes (got {n_groups}) for a grouped "
            "train/test split."
        )

    is_classification = (target == "action_argmax")

    if is_classification:
        classes, counts = np.unique(y, return_counts=True)
        majority_frac = counts.max() / counts.sum()
        majority_baseline = 1.0 / len(classes)   # balanced accuracy of a
                                                   # majority-class-only model
        print(f"Class distribution for '{target}': "
              f"{dict(zip(classes.tolist(), counts.tolist()))}")
        print(f"Majority class = {classes[np.argmax(counts)]} "
              f"({majority_frac*100:.1f}% of samples). "
              f"Majority-only balanced-accuracy floor = {majority_baseline:.3f}")
        if majority_frac > 0.7:
            print(f"  WARNING: severe class imbalance ({majority_frac*100:.0f}% "
                  "one class) — a probe that collapses to predicting only the "
                  "majority class will land exactly at the floor above for "
                  "EVERY fold. If both scores below equal that floor, neither "
                  "probe learned anything — that's not the same conclusion as "
                  "'h has no advantage over phy_state'.")

    gkf = GroupKFold(n_splits=n_splits_eff)

    def _score_fold(X, y, train_idx, test_idx):
        if is_classification:
            # LogisticRegression with class_weight='balanced' instead of
            # RidgeClassifier: Ridge's one-hot least-squares classification
            # collapses to the majority class far more readily under class
            # imbalance / weak signal, which would hide a real-but-subtle
            # effect rather than just correctly reporting "no effect".
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1.0 / alpha, class_weight="balanced",
                                         max_iter=1000)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            return balanced_accuracy_score(y[test_idx], pred)
        else:
            model = Ridge(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            return r2_score(y[test_idx], pred)

    phy_folds, h_folds = [], []
    for train_idx, test_idx in gkf.split(X_h, y, groups=groups):
        phy_folds.append(_score_fold(X_phy, y, train_idx, test_idx))
        h_folds.append(_score_fold(X_h, y, train_idx, test_idx))

    metric_name = "balanced accuracy" if is_classification else "R²"
    print(f"H2 — predicting next-step '{target}' ({metric_name}):")
    print(f"  phy_state probe : {np.mean(phy_folds):+.3f}  (folds: {np.round(phy_folds, 3)})")
    print(f"  h (trunk) probe : {np.mean(h_folds):+.3f}  (folds: {np.round(h_folds, 3)})")
    diff = np.mean(h_folds) - np.mean(phy_folds)
    print(f"  difference (h - phy_state): {diff:+.3f}  "
          f"{'<-- h adds predictive info beyond phy_state' if diff > 0.02 else '(no clear advantage)'}")

    result = {
        "phy_state_score": float(np.mean(phy_folds)),
        "phy_state_folds": phy_folds,
        "h_score": float(np.mean(h_folds)),
        "h_folds": h_folds,
        "metric": metric_name,
    }
    if is_classification:
        result["majority_baseline"] = majority_baseline
        both_at_floor = (abs(np.mean(phy_folds) - majority_baseline) < 0.01 and
                          abs(np.mean(h_folds) - majority_baseline) < 0.01)
        if both_at_floor:
            print("  NOTE: both probes are at the majority-class floor — "
                  "neither learned a usable signal for this target. Consider "
                  "more training episodes, a non-linear probe, or a coarser "
                  "target (e.g. binary 'ate vs not') before concluding "
                  "anything about h vs phy_state from this specific test.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Unified entry point: pick the layer, pick raw-vs-PCA, pick how many
# components/dimensions to look at, get the plot. This wraps the existing
# capture_* / run_pca / plot_* functions so each of the three choices is one
# explicit argument instead of remembering which combination of function
# calls produces what.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_signature(agent, env, network="actor", signature="combined",
                          mode="pca", n_components=3, plot_type="lines",
                          n_episodes=1, deterministic=False, save_path=None,
                          analyze_shadow=True, n_episodes_for_shadow_stats=20):
    """
    One entry point covering all three choices you asked for:

      1. WHICH LAYER  -> network ("actor"/"critic") x signature
                          ("combined"/"phy"/"food"). Same options as
                          capture_trunk_episode — see that docstring for
                          which combinations require separate_branches=True.

      2. RAW OR PCA   -> mode="raw" plots the bottleneck dimensions directly,
                          no PCA at all (only sensible when the bottleneck
                          is already small, e.g. 8 dims — for a 256-dim
                          no-bottleneck trunk, request a small n_components
                          and mode="pca" instead, or you'll get an
                          unreadable 256-panel grid).
                          mode="pca" runs PCA first, plots the components.

      3. HOW MANY     -> n_components: for mode="pca", how many principal
                          components to extract and show. For mode="raw",
                          how many of the raw dimensions to show (first
                          n_components of however many the layer has) —
                          use n_components=None to show ALL raw dimensions.

    plot_type : "lines" (stacked line plot, n_components/dims each as a row)
                "grid"  (small-multiples grid — better for >3 components/dims)
                "3d"    (3D trajectory; requires exactly 3 components/dims —
                         only valid combined with n_components=3)

    For mode="raw" with plot_type="lines"/"grid", this calls
    plot_pca_lines/plot_pca_grid directly on the raw z (those functions
    don't actually care whether their input came from PCA or not — they
    just plot whatever (T, dim) array you hand them).

    Shadow nutrient analysis (fullness/hunger, or whatever
    env_config.SHADOW_NUTRIENT_CONFIG defines) — runs automatically
    whenever env.shadow_nutrient_names is non-empty, on top of whichever
    network/signature/mode/n_components you already picked above:

      analyze_shadow              : set False to skip this section entirely
                                     (e.g. if the env has no shadow nutrients
                                     configured, this is a no-op either way).
      n_episodes_for_shadow_stats : the held-out decodability test
                                     (per_shadow_decodability) needs
                                     GroupKFold over multiple episodes —
                                     it draws its OWN pool of episodes via
                                     capture_trunk_multi_episode, separate
                                     from whatever n_episodes you set above
                                     for the main plot. This is printed
                                     explicitly so it's clear the stats and
                                     the plotted trajectory are not
                                     necessarily the same episode(s).

    What this section does, in order:
      (a) plot_shadow_trends   — line+scatter trend of each shadow nutrient
                                  over the SAME episode(s) already captured
                                  for the main plot above (same eating-point
                                  markers, same sleep shading).
      (b) shadow_cross_correlation — Pearson r between shadow nutrients
                                  themselves (e.g. fullness vs hunger) on
                                  that same episode — a sanity check on the
                                  reconstructed trace, independent of z.
      (c) per_shadow_decodability — held-out R² of the FULL, untruncated
                                  captured representation (all `dim` raw
                                  dims — NOT the `k`-dim slice used for the
                                  plot above) predicting each shadow
                                  nutrient's RAW LEVEL, via its own
                                  multi-episode capture. This is
                                  deliberately decoupled from `mode`/
                                  `n_components`: those control what gets
                                  PLOTTED above, and previously also
                                  silently truncated this stats section to
                                  the same k dims (e.g. n_components=2 for
                                  a readable line plot meant decodability
                                  was only ever tested against 2 of 16
                                  z_phy dims) — a full PCA transform would
                                  give identical held-out R² to the full
                                  raw representation anyway (PCA without
                                  whitening is an orthogonal rotation, and
                                  Ridge is rotation-invariant under that),
                                  so using the untransformed h_stats
                                  directly loses nothing and removes the
                                  coupling. See per_shadow_decodability's
                                  docstring for why this isn't directly
                                  comparable to H1's deficit-based R².

    Returns
    -------
    (h_or_pcs, meta, fig_or_pca_object, shadow_results)
        shadow_results : dict with keys 'cross_corr' (DataFrame or None) and
                          'decodability' (DataFrame or None) — None for
                          either if analyze_shadow=False or the env has no
                          shadow nutrients configured.
    """
    if n_episodes == 1:
        h_seq, meta = capture_trunk_episode(
            agent, env, deterministic=deterministic, network=network,
            signature=signature,
        )
    else:
        h_seq, meta = capture_trunk_multi_episode(
            agent, env, n_episodes=n_episodes, deterministic=deterministic,
            network=network, signature=signature,
        )
        print(f"NOTE: n_episodes={n_episodes} pools multiple episodes into one "
              f"array for plotting — line/3D plots will show discontinuous "
              f"jumps at episode boundaries. Use n_episodes=1 for a clean "
              f"single-episode trajectory view, or use this only with "
              f"plot_type='grid' for a denser but still readable view.")

    dim = h_seq.shape[1]

    if mode == "raw":
        k = dim if n_components is None else min(n_components, dim)
        data = h_seq[:, :k]
        pca_obj = None
        print(f"Showing RAW dimensions 0..{k-1} of {dim} "
              f"(network={network!r}, signature={signature!r}) — no PCA applied.")
    elif mode == "pca":
        k = min(n_components, dim)
        if k < dim:
            data, pca_obj = run_pca(h_seq, n_components=k)
        else:
            data, pca_obj = run_pca(h_seq, n_components=dim)
    else:
        raise ValueError(f"mode must be 'raw' or 'pca', got {mode!r}")

    if plot_type == "3d":
        if data.shape[1] != 3:
            raise ValueError(
                f"plot_type='3d' requires exactly 3 components/dims, got "
                f"{data.shape[1]} (n_components={n_components}). Set "
                "n_components=3 for a 3D plot."
            )
        fig = plot_pca_3d(data, meta, agent.is_continuous, save_path=save_path,
                            dim_label=("PC" if mode == "pca" else "dim"))
    elif plot_type == "lines":
        dim_label = "PC" if mode == "pca" else "dim"
        fig = plot_pca_lines(data, meta, agent.is_continuous, save_path=save_path,
                               dim_label=dim_label)
    elif plot_type == "grid":
        # plot_pca_grid re-runs PCA internally if called the normal way, but
        # we've already done the raw/PCA choice above — call the lower-level
        # plotting logic directly on `data` to avoid computing PCA twice or
        # silently overriding mode="raw".
        fig = _plot_grid_raw(data, meta, agent.is_continuous, save_path=save_path)
    else:
        raise ValueError(f"plot_type must be 'lines', 'grid', or '3d', got {plot_type!r}")

    # ── Shadow nutrient analysis (fullness/hunger, etc.) ───────────────────
    # Runs against whatever network/signature/mode/n_components was already
    # selected above. No-op if the env has no shadow nutrients configured,
    # or if analyze_shadow=False.
    shadow_results = {"cross_corr": None, "decodability": None,
                       "corr_mse_df": None}
    shadow_names = list(getattr(env, "shadow_nutrient_names", []))

    if analyze_shadow and shadow_names:
        print(f"\n── Shadow nutrient analysis: {shadow_names} "
              f"(network={network!r}, signature={signature!r}, mode={mode!r}) "
              "── observation/reward were never touched by these during "
              "training; this is post-hoc only.\n")

        # (a) Trend plot — same episode(s) as the main plot above, so the
        # eating-point markers correspond to the same trajectory.
        # Insert the "_shadow" suffix BEFORE the file extension, not after —
        # appending it after (e.g. "plot.png" -> "plot.png_shadow") breaks
        # matplotlib's format-from-extension detection.
        if save_path:
            root, ext = os.path.splitext(save_path)
            shadow_save_path = f"{root}_shadow{ext}" if ext else f"{save_path}_shadow.png"
        else:
            shadow_save_path = None
        plot_shadow_trends(meta, agent.is_continuous, shadow_names=shadow_names,
                             save_path=shadow_save_path)

        # (b) Cross-correlation between shadow nutrients themselves.
        shadow_results["cross_corr"] = shadow_cross_correlation(meta, shadow_names=shadow_names)

        # (c) Held-out decodability of each shadow nutrient from `data`.
        # Needs its OWN multi-episode pool for GroupKFold — separate from
        # whatever n_episodes was used for the plot above.
        print(f"\nRunning shadow decodability over a SEPARATE pool of "
              f"{n_episodes_for_shadow_stats} episodes (independent of the "
              f"{n_episodes} episode(s) plotted above) ...")
        h_stats, meta_stats = capture_trunk_multi_episode(
            agent, env, n_episodes=n_episodes_for_shadow_stats,
            deterministic=deterministic, network=network, signature=signature,
        )
        # Deliberately NOT sliced to [:, :k] or PCA-reduced to k components
        # — this stats section always uses the FULL captured representation,
        # independent of whatever mode/n_components was chosen for the plot
        # above. Previously this reused `k`, so e.g. N_COMPONENTS=2 (chosen
        # only to keep the line plot readable) silently also capped these
        # stats at the first 2 of `dim` raw dimensions. A full PCA
        # transform would give identical held-out R² to the full raw array
        # anyway (PCA without whitening is an orthogonal rotation; Ridge is
        # rotation-invariant under that), so there's no accuracy trade-off
        # in always using h_stats as captured — only a bug removed.
        data_stats = h_stats
        stats_dim_label = "dim"   # always raw-z labeling; data_stats is never PCA-reduced

        shadow_results["decodability"] = per_shadow_decodability(
            data_stats, meta_stats, shadow_names=shadow_names,
        )

        # (d) Per-dim correlation + min-max-normalised MSE screen, on the
        # SAME data_stats pool used for decodability above (descriptive
        # screen, not a held-out claim — same caveat as
        # compute_z_phy_shadow_correlations / per_pc_deficit_correlation).
        shadow_targets = {n: meta_stats[n] for n in shadow_names}
        shadow_results["corr_mse_df"] = compute_data_shadow_correlations(
            data_stats, shadow_targets, dim_label=stats_dim_label,
        )
        plot_data_shadow_correlation_heatmap(
            shadow_results["corr_mse_df"], value="r",
            title=f"{stats_dim_label} vs shadow-nutrient correlation (r)  "
                  f"[full {data_stats.shape[1]}-dim representation]",
            save_path=_path_suffix(save_path, "_shadow_corr_heatmap"),
        )
        plot_data_shadow_correlation_heatmap(
            shadow_results["corr_mse_df"], value="mse",
            title=f"{stats_dim_label} vs shadow-nutrient MSE (min-max normalised)  "
                  f"[full {data_stats.shape[1]}-dim representation]",
            save_path=_path_suffix(save_path, "_shadow_mse_heatmap"),
        )

    return data, meta, (pca_obj if mode == "pca" else fig), shadow_results


def _plot_grid_raw(data, meta, is_continuous, save_path=None):
    """
    Grid plot for already-prepared (T, k) data (raw dims OR pca components —
    caller decides), without plot_pca_grid's internal PCA call. Mirrors
    plot_pca_grid's layout exactly, just skipping the "fit PCA" step.
    """
    k = data.shape[1]
    colors, handles = _action_color_setup(meta, is_continuous)
    T = data.shape[0]
    t = np.arange(T)

    ncols = 2
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.0 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i in range(k):
        ax = axes[i]
        ax.plot(t, data[:, i], color="steelblue", lw=0.9, alpha=0.6, zorder=1)
        ax.scatter(t, data[:, i], c=colors, s=10, zorder=2, edgecolors="none")
        ax.set_ylabel(f"dim {i}", fontsize=8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for i in range(k, len(axes)):
        axes[i].axis("off")

    axes[0].legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    fig.suptitle(f"{k} raw dimensions over one episode", y=1.0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# z_phy <-> shadow nutrient (hunger / fullness) analysis
#
# Everything below asks one focused question: does the phy-branch bottleneck
# z_phy (signature="phy", requires the agent to have been built with
# separate_branches=True) carry a hunger/fullness-like signature, given that
# fullness/hunger were NEVER part of the observation or reward (see
# env_config.SHADOW_NUTRIENT_CONFIG, env.py's shadow-nutrient machinery).
#
# Design choices, stated once here rather than repeated in every docstring:
#   - Capture and analysis are kept apart from plotting throughout, per file
#     convention: capture_* / compute_* / decode_* / shuffle_* return data
#     (ndarrays, dicts, DataFrames); plot_* functions take that data as input
#     and never call capture/compute themselves.
#   - All cross-validated decoding is GroupKFold'd by episode_idx, exactly
#     like test_h1_deficit_decodability / test_h2_action_predictivity above
#     — rows within an episode are autocorrelated, so a random split would
#     leak information across train/test and inflate R².
#   - Sleep/wake state is deliberately NOT a variable of interest anywhere
#     in this section. It still exists in phy_state / the environment, but
#     none of the functions below condition on it, color by it, or report
#     on it — that's intentionally out of scope here.
#   - Language: avoid "hunger neuron" framing anywhere a single dimension
#     is discussed. A dimension that correlates with hunger is reported as
#     "hunger-associated", not as a discovered unit with that function —
#     correlation across one episode or one probe does not establish that
#     a single coordinate of z_phy is dedicated to encoding hunger. The
#     full-vector decoding tests (decode_shadow_from_z_phy /
#     compare_real_vs_shuffled_z_phy_decoding) are the load-bearing
#     evidence; the per-dimension correlation screen is a cheap first look,
#     not a conclusion (same posture as per_pc_deficit_correlation above).
# ══════════════════════════════════════════════════════════════════════════════


def capture_phy_bottleneck_shadow_data(agent, env, n_episodes=20,
                                         deterministic=False, max_steps=None,
                                         network="actor", shadow_names=None):
    """
    Capture z_phy (the phy-branch bottleneck) and shadow-nutrient targets
    (e.g. hunger, fullness, cck, ghrelin, glp_1, pyy — whatever is
    configured) over multiple episodes, pooled for analysis.

    Thin wrapper around capture_trunk_multi_episode(..., signature="phy")
    that additionally validates the requested shadow nutrients are present
    and repackages them as a separate `targets` dict so downstream
    functions don't need to know about meta's other keys.

    Parameters
    ----------
    agent         : PPOAgent — must have been built with separate_branches=True
                    (otherwise there is no z_phy to read; see _read_z).
    env           : FoodEnv — must have every name in `shadow_names`
                    configured in env_config.SHADOW_NUTRIENT_CONFIG (i.e.
                    env.shadow_nutrient_names must be a superset).
    n_episodes    : int — number of episodes to pool (passed straight to
                    capture_trunk_multi_episode). More episodes -> more
                    independent groups -> more stable held-out estimates
                    in the decoding functions below.
    deterministic : bool — passed to agent.act() inside the rollout. Keep
                    this consistent with whatever else you're comparing
                    against (see this module's header docstring).
    max_steps     : optional per-episode step cap (defaults to env.max_steps).
    network       : "actor" (default) or "critic". "critic" only works if
                    agent.args['shared'] is False — see _get_trunk_module.
    shadow_names  : list of shadow-nutrient names to include as targets, or
                    None (default) to use ALL of env.shadow_nutrient_names
                    — i.e. whatever is currently configured in
                    env_config.SHADOW_NUTRIENT_CONFIG. Pass an explicit
                    subset (e.g. ["hunger", "fullness"]) to restrict the
                    analysis to fewer nutrients than the env has loaded.

    Returns
    -------
    z_phy       : (N, z_dim) float32 ndarray — pooled phy-branch bottleneck
                  activations, N = sum of per-episode step counts.
    targets     : dict with one key per name in `shadow_names`, each a (N,)
                  float32 ndarray, row-aligned with z_phy.
    episode_idx : (N,) int32 ndarray — which pooled episode each row came
                  from (0-indexed), for GroupKFold grouping downstream.
    meta        : the full meta_all dict from capture_trunk_multi_episode,
                  returned as-is in case the caller wants other fields
                  (action_sum, phy_state, reward, etc.).

    Raises
    ------
    ValueError
        If shadow_names is explicitly [] (nothing to analyze), if any
        requested name is not present in env.shadow_nutrient_names (and
        therefore not in the captured meta), or if z_phy could not be read
        (e.g. agent was not built with separate_branches=True — this
        surfaces as whatever _read_z already raises, not swallowed here).
    """
    available = list(getattr(env, "shadow_nutrient_names", []))
    if shadow_names is None:
        shadow_names = available
    if not shadow_names:
        raise ValueError(
            "No shadow nutrients to analyze — env.shadow_nutrient_names is "
            f"empty (configure at least one in env_config.SHADOW_NUTRIENT_CONFIG), "
            "or an empty shadow_names list was passed explicitly."
        )

    missing_shadow = set(shadow_names) - set(available)
    if missing_shadow:
        raise ValueError(
            f"env is missing shadow nutrient(s) {sorted(missing_shadow)} — "
            "capture_phy_bottleneck_shadow_data requires every name in "
            "shadow_names to be configured in env_config.SHADOW_NUTRIENT_CONFIG. "
            f"Found: {sorted(available)}."
        )

    z_phy, meta = capture_trunk_multi_episode(
        agent, env, n_episodes=n_episodes, deterministic=deterministic,
        max_steps=max_steps, network=network, signature="phy",
    )

    for name in shadow_names:
        if name not in meta:
            raise ValueError(
                f"'{name}' was not found in captured meta even though "
                f"env.shadow_nutrient_names included it — this points at a "
                "bug in capture_trunk_episode's shadow-nutrient bookkeeping, "
                "not a misconfiguration here."
            )

    targets = {name: np.asarray(meta[name], dtype=np.float32) for name in shadow_names}
    episode_idx = np.asarray(meta["episode_idx"], dtype=np.int32)

    return z_phy, targets, episode_idx, meta


def _path_suffix(save_path, suffix):
    """Insert `suffix` before the file extension of save_path, or return
    None if save_path is None. Mirrors the logic visualize_signature
    already used inline for its '_shadow' suffix, lifted out so the new
    correlation/MSE heatmap calls can reuse it."""
    if not save_path:
        return None
    root, ext = os.path.splitext(save_path)
    return f"{root}{suffix}{ext}" if ext else f"{save_path}{suffix}.png"


def compute_data_shadow_correlations(data, targets, method="pearson", dim_label="dim"):
    """
    Generic per-dimension correlation + min-max-normalised MSE screen
    between ANY (T, D) array (raw bottleneck dims, PCA components, etc.)
    and shadow targets (hunger, fullness, ...). This is the engine behind
    compute_z_phy_shadow_correlations (which is a thin wrapper calling this
    with dim_label="z_phy"); use this version directly when probing PCs or
    raw-mode dims from visualize_signature, where the natural label is
    "PC"/"dim" rather than "z_phy".

    Same caveats as compute_z_phy_shadow_correlations: descriptive screen
    on pooled, autocorrelated rows — not a held-out claim. Correlation is
    scale-invariant (computed on raw data/targets); MSE requires the
    min-max normalisation step to be comparable across dims/targets with
    different native ranges.

    Parameters
    ----------
    data       : (T, D) ndarray.
    targets    : dict[str, (T,) ndarray].
    method     : "pearson" or "spearman".
    dim_label  : column-label prefix for the returned 'dim_name' column,
                 e.g. "PC" -> "PC1", "PC2", ...; "dim" -> "dim0", "dim1", ...

    Returns
    -------
    pandas.DataFrame with columns: target, latent_dim, dim_name, r, p,
    abs_r, mse. One row per (target, dim) pair, sorted by abs_r descending.
    """
    import pandas as pd

    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    if method == "pearson":
        from scipy.stats import pearsonr as _corr_fn
    else:
        from scipy.stats import spearmanr as _corr_fn

    data = np.asarray(data)
    n_dims = data.shape[1]
    data_norm = _minmax_normalize(data, axis=0)

    rows = []
    for target_name, y in targets.items():
        y = np.asarray(y)
        y_norm = _minmax_normalize(y, axis=0)
        for j in range(n_dims):
            r, p = _corr_fn(data[:, j], y)
            mse = float(np.mean((data_norm[:, j] - y_norm) ** 2))
            rows.append({
                "target": target_name,
                "latent_dim": j,
                "dim_name": f"{dim_label}{j+1}" if dim_label == "PC" else f"{dim_label}{j}",
                "r": float(r), "p": float(p), "abs_r": float(abs(r)),
                "mse": mse,
            })

    df = pd.DataFrame(rows, columns=["target", "latent_dim", "dim_name", "r", "p", "abs_r", "mse"])
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    print(
        f"[compute_data_shadow_correlations] {method} correlations + "
        f"min-max-normalised MSE over {data.shape[0]} pooled rows "
        f"({dim_label} dims vs {list(targets.keys())}). Descriptive screen "
        "on autocorrelated rows — use per_shadow_decodability (held-out "
        "R², GroupKFold by episode) for the actual evidence."
    )
    return df


def plot_data_shadow_correlation_heatmap(corr_mse_df, value="r", title=None, save_path=None):
    """
    Generic heatmap for compute_data_shadow_correlations' output. Set
    value="r" for the correlation heatmap (diverging RdBu_r colormap,
    sign visible) or value="mse" for the MSE heatmap (sequential YlOrRd,
    lower=better, no sign). Rows = targets, cols = dim_name.

    Parameters
    ----------
    corr_mse_df : pandas.DataFrame — output of compute_data_shadow_correlations
                  (columns include target, latent_dim, dim_name, r, mse).
    value       : "r" or "mse" — which column to plot.
    title       : optional plot title; a sensible default is used if None.
    save_path   : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if value not in ("r", "mse"):
        raise ValueError(f"value must be 'r' or 'mse', got {value!r}")

    pivot = corr_mse_df.pivot(index="target", columns="latent_dim", values=value)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    # Use the dim_name labels (e.g. PC1, PC2, dim0, dim1) rather than the
    # raw integer latent_dim index for x-tick labels.
    label_lookup = corr_mse_df.drop_duplicates("latent_dim").set_index("latent_dim")["dim_name"]
    pivot.columns = [label_lookup.get(c, str(c)) for c in pivot.columns]

    if value == "r":
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1]), max(3, 0.6 * pivot.shape[0] + 1.5)))
        vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title(title or "Correlation (r)", fontsize=12)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > 0.6 * vmax else "black")
        fig.colorbar(im, ax=ax, label="Pearson/Spearman r", fraction=0.04, pad=0.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig
    else:
        return _plot_mse_heatmap(pivot, title=title or "MSE (min-max normalised)",
                                   save_path=save_path)


def compute_z_phy_shadow_correlations(z_phy, targets, method="pearson"):
    """
    Per-dimension correlation AND min-max-normalised MSE screen: for each
    column of z_phy, compute its correlation with each shadow target
    (hunger, fullness), and the MSE between the two after independently
    min-max normalising z_phy[:, j] and the target to [0, 1] (see
    _minmax_normalize). Correlation is scale-invariant so it's computed on
    the raw values as before; MSE is not, so without the normalisation step
    the MSE numbers would mix nutrients/dims that live on very different
    native scales and not be comparable to each other.

    This is a cheap first look, NOT evidence on its own — see this module's
    section header. p-values are computed treating rows as independent,
    which they are NOT (consecutive rows within an episode are strongly
    autocorrelated). That inflates significance: with thousands of
    autocorrelated rows, p can be tiny even for a correlation that would
    not survive an honest episode-grouped test. Use the p column only to
    rank candidate dimensions for a closer look, never to claim
    significance — test_z_phy_shadow_decodability-style held-out R²
    (decode_shadow_from_z_phy / compare_real_vs_shuffled_z_phy_decoding) is
    the evidence that matters. The same caveat applies to MSE here: a low
    MSE on this same pooled, autocorrelated data is descriptive, not a
    held-out claim.

    Parameters
    ----------
    z_phy   : (N, z_dim) ndarray — any z_dim, not hard-coded.
    targets : dict[str, (N,) ndarray] — e.g. {"hunger": ..., "fullness": ...}.
    method  : "pearson" or "spearman".

    Returns
    -------
    pandas.DataFrame with columns: target, latent_dim, r, p, abs_r, mse
        mse = MSE(min-max normalised z_phy[:, j], min-max normalised target)
              — lower is better, not directly comparable in magnitude to r.
    One row per (target, latent_dim) pair, sorted by abs_r descending.
    """
    import pandas as pd

    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    if method == "pearson":
        from scipy.stats import pearsonr as _corr_fn
    else:
        from scipy.stats import spearmanr as _corr_fn

    z_dim = z_phy.shape[1]

    # Normalised once up front (not per-row in the loop) — each z_phy
    # column and each target independently min-max scaled to [0, 1].
    z_phy_norm = _minmax_normalize(z_phy, axis=0)
    targets_norm = {name: _minmax_normalize(np.asarray(y), axis=0)
                     for name, y in targets.items()}

    rows = []
    for target_name, y in targets.items():
        y = np.asarray(y)
        y_norm = targets_norm[target_name]
        for j in range(z_dim):
            r, p = _corr_fn(z_phy[:, j], y)
            mse = float(np.mean((z_phy_norm[:, j] - y_norm) ** 2))
            rows.append({
                "target": target_name,
                "latent_dim": j,
                "r": float(r),
                "p": float(p),
                "abs_r": float(abs(r)),
                "mse": mse,
            })

    df = pd.DataFrame(rows, columns=["target", "latent_dim", "r", "p", "abs_r", "mse"])
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    n_total = z_phy.shape[0]
    print(
        f"[compute_z_phy_shadow_correlations] {method} correlations + "
        f"min-max-normalised MSE over {n_total} pooled rows. NOTE: "
        "p-values above assume independent rows; rows within an episode "
        "are autocorrelated, so these p-values are optimistic "
        "(anti-conservative) and should be used only to rank candidate "
        "dimensions, not to claim significance. The same applies to MSE — "
        "both are descriptive screens on the same pooled, autocorrelated "
        "data. Use decode_shadow_from_z_phy / "
        "compare_real_vs_shuffled_z_phy_decoding for an honest, "
        "episode-grouped held-out estimate."
    )

    return df


def shuffle_z_within_episodes(z_phy, episode_idx, seed=0):
    """
    Shuffle rows of z_phy WITHIN each episode (never across episodes),
    producing a null-control array that preserves the marginal distribution
    of z_phy (same set of row vectors, same per-episode group sizes) while
    destroying the timestep-to-timestep alignment between z_phy and any
    target (hunger, fullness, etc.) that varies smoothly over time.

    Used as the control condition in compare_real_vs_shuffled_z_phy_decoding:
    if decoding accuracy on shuffled z_phy is comparable to real z_phy,
    real decoding was likely picking up coincidental drift/autocorrelation
    rather than a genuine timestep-aligned relationship.

    Parameters
    ----------
    z_phy       : (N, z_dim) ndarray.
    episode_idx : (N,) int array — same length as z_phy, episode group label
                  per row (as returned by capture_phy_bottleneck_shadow_data).
    seed        : int — RNG seed, for reproducible shuffles.

    Returns
    -------
    z_phy_shuffled : (N, z_dim) ndarray — same shape and dtype as z_phy.
    """
    rng = np.random.default_rng(seed)
    z_phy_shuffled = np.array(z_phy, copy=True)

    for ep in np.unique(episode_idx):
        mask = episode_idx == ep
        idx_in_ep = np.flatnonzero(mask)
        permuted = rng.permutation(idx_in_ep)
        z_phy_shuffled[idx_in_ep] = z_phy[permuted]

    return z_phy_shuffled


def decode_shadow_from_z_phy(z_phy, targets, episode_idx, alpha=1.0, n_splits=5):
    """
    Cross-validated Ridge decoding of each shadow target from the FULL
    z_phy vector, using GroupKFold grouped by episode_idx (same machinery
    as _episode_group_kfold_r2 / test_h1_deficit_decodability above — rows
    within an episode are autocorrelated, so grouping by episode prevents
    train/test leakage).

    Works on whatever is passed as z_phy — real or shuffled (e.g. pass the
    output of shuffle_z_within_episodes to get the null-control numbers
    compare_real_vs_shuffled_z_phy_decoding needs).

    Parameters
    ----------
    z_phy       : (N, z_dim) ndarray — any z_dim.
    targets     : dict[str, (N,) ndarray] — e.g. {"hunger": ..., "fullness": ...}.
    episode_idx : (N,) int array — episode group label per row.
    alpha       : Ridge regularisation strength.
    n_splits    : requested number of folds (capped to the number of unique
                  episodes if fewer are available).

    Returns
    -------
    dict[target_name] -> {
        "mean_r2"     : float — mean held-out R² across folds,
        "fold_r2"     : list[float] — per-fold held-out R²,
        "predictions" : (N,) ndarray — concatenated out-of-fold predictions,
                        aligned to the ORIGINAL row order of z_phy/targets
                        (every row is predicted exactly once, by whichever
                        fold held it out),
        "true"        : (N,) ndarray — the corresponding true values, same
                        order as "predictions" (i.e. just targets[name],
                        included here so plotting functions don't need to
                        also thread the original targets dict through).
    }
    """
    n_groups = len(np.unique(episode_idx))
    n_splits_eff = min(n_splits, n_groups)
    if n_splits_eff < 2:
        raise ValueError(
            f"Need at least 2 episodes (got {n_groups}) for a grouped "
            "train/test split — increase n_episodes in "
            "capture_phy_bottleneck_shadow_data."
        )

    gkf = GroupKFold(n_splits=n_splits_eff)
    results = {}

    for target_name, y in targets.items():
        y = np.asarray(y)
        oof_pred = np.full_like(y, np.nan, dtype=np.float64)
        fold_r2 = []

        for train_idx, test_idx in gkf.split(z_phy, y, groups=episode_idx):
            model = Ridge(alpha=alpha)
            model.fit(z_phy[train_idx], y[train_idx])
            pred = model.predict(z_phy[test_idx])
            oof_pred[test_idx] = pred
            fold_r2.append(r2_score(y[test_idx], pred))

        results[target_name] = {
            "mean_r2": float(np.mean(fold_r2)),
            "fold_r2": fold_r2,
            "predictions": oof_pred,
            "true": y,
        }
        print(f"decode_shadow_from_z_phy [{target_name:8s}] held-out R² = "
              f"{np.mean(fold_r2):+.3f}  (folds: {np.round(fold_r2, 3)})")

    return results


def compare_real_vs_shuffled_z_phy_decoding(z_phy, targets, episode_idx,
                                              alpha=1.0, n_splits=5, seed=0):
    """
    Run decode_shadow_from_z_phy on real z_phy and on a within-episode
    shuffled control (shuffle_z_within_episodes), then assemble a tidy
    comparison. This is the load-bearing test for "does z_phy carry a
    timestep-aligned hunger/fullness signature, or would a representation
    with the same marginal distribution but no temporal alignment do just
    as well" — a real result should show a clear gap between conditions,
    not just a positive real-z_phy R² in isolation (a positive R² alone
    could come from slow co-drift rather than genuine moment-to-moment
    tracking).

    Parameters
    ----------
    z_phy, targets, episode_idx, alpha, n_splits : see decode_shadow_from_z_phy.
    seed : int — passed to shuffle_z_within_episodes for reproducibility.

    Returns
    -------
    comparison_df : pandas.DataFrame with columns:
        target, condition ("real_z_phy" or "shuffled_z_phy"), mean_r2, fold_r2
        (fold_r2 stored as a list per row).
    predictions : dict[(target_name, condition)] -> {"predictions": ..., "true": ...}
        from the underlying decode_shadow_from_z_phy calls, in case the
        caller wants predicted-vs-true plots for either condition.
    """
    import pandas as pd

    z_phy_shuffled = shuffle_z_within_episodes(z_phy, episode_idx, seed=seed)

    real_results = decode_shadow_from_z_phy(
        z_phy, targets, episode_idx, alpha=alpha, n_splits=n_splits,
    )
    shuffled_results = decode_shadow_from_z_phy(
        z_phy_shuffled, targets, episode_idx, alpha=alpha, n_splits=n_splits,
    )

    rows = []
    predictions = {}
    for target_name in targets:
        rows.append({
            "target": target_name,
            "condition": "real_z_phy",
            "mean_r2": real_results[target_name]["mean_r2"],
            "fold_r2": real_results[target_name]["fold_r2"],
        })
        rows.append({
            "target": target_name,
            "condition": "shuffled_z_phy",
            "mean_r2": shuffled_results[target_name]["mean_r2"],
            "fold_r2": shuffled_results[target_name]["fold_r2"],
        })
        predictions[(target_name, "real_z_phy")] = {
            "predictions": real_results[target_name]["predictions"],
            "true": real_results[target_name]["true"],
        }
        predictions[(target_name, "shuffled_z_phy")] = {
            "predictions": shuffled_results[target_name]["predictions"],
            "true": shuffled_results[target_name]["true"],
        }

    comparison_df = pd.DataFrame(rows, columns=["target", "condition", "mean_r2", "fold_r2"])

    for target_name in targets:
        real_r2 = real_results[target_name]["mean_r2"]
        shuf_r2 = shuffled_results[target_name]["mean_r2"]
        gap = real_r2 - shuf_r2
        print(f"compare_real_vs_shuffled [{target_name:8s}] "
              f"real={real_r2:+.3f}  shuffled={shuf_r2:+.3f}  gap={gap:+.3f}  "
              f"{'<-- real z_phy clearly beats the shuffled control' if gap > 0.02 else '(no clear gap)'}")

    return comparison_df, predictions


# ──────────────────────────────────────────────────────────────────────────────
# Plotting functions — take analysis-function output as input, never call
# capture/compute/decode themselves (per this module's separation rule).
# ──────────────────────────────────────────────────────────────────────────────

def plot_z_phy_shadow_correlation_heatmap(corr_df, save_path=None):
    """
    Heatmap of correlation coefficients from compute_z_phy_shadow_correlations.

    Rows = targets (hunger, fullness, ...); columns = latent dims
    (z_phy_0, z_phy_1, ...). Cell value = r (not abs_r), so sign is visible.

    Parameters
    ----------
    corr_df   : pandas.DataFrame — output of compute_z_phy_shadow_correlations
                (columns: target, latent_dim, r, p, abs_r, mse).
    save_path : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = corr_df.pivot(index="target", columns="latent_dim", values="r")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    col_labels = [f"z_phy_{j}" for j in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(col_labels)), max(3, 0.6 * len(pivot.index) + 1.5)))
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    vmax = max(vmax, 1e-6)

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("z_phy vs shadow-nutrient correlation (r)", fontsize=12)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 0.6 * vmax else "black")

    fig.colorbar(im, ax=ax, label="Pearson/Spearman r", fraction=0.04, pad=0.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_z_phy_shadow_mse_heatmap(corr_df, save_path=None):
    """
    Heatmap of min-max-normalised MSE from compute_z_phy_shadow_correlations'
    'mse' column. Same row/column layout as
    plot_z_phy_shadow_correlation_heatmap (rows = targets, cols = z_phy_j)
    so the two can be compared side by side — but uses a sequential
    colormap (darker = worse/higher error) since, unlike correlation,
    there's no meaningful sign here.

    Parameters
    ----------
    corr_df   : pandas.DataFrame — output of compute_z_phy_shadow_correlations
                (columns: target, latent_dim, r, p, abs_r, mse).
    save_path : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = corr_df.pivot(index="target", columns="latent_dim", values="mse")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    pivot.columns = [f"z_phy_{j}" for j in pivot.columns]
    return _plot_mse_heatmap(
        pivot, title="z_phy vs shadow-nutrient MSE (min-max normalised)",
        save_path=save_path,
    )


def plot_z_phy_decoding_comparison(decoding_df, save_path=None):
    """
    Bar plot comparing real-z_phy vs shuffled-z_phy held-out R² for each
    target, with individual fold values overlaid as scatter points so the
    spread across folds is visible alongside the mean bar.

    Parameters
    ----------
    decoding_df : pandas.DataFrame — output of
                  compare_real_vs_shuffled_z_phy_decoding (columns: target,
                  condition, mean_r2, fold_r2).
    save_path   : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    targets_order = list(decoding_df["target"].unique())
    conditions = ["real_z_phy", "shuffled_z_phy"]
    colors = {"real_z_phy": "steelblue", "shuffled_z_phy": "lightcoral"}

    fig, ax = plt.subplots(figsize=(max(5, 2.2 * len(targets_order)), 5))

    bar_width = 0.35
    x = np.arange(len(targets_order))

    for c_i, condition in enumerate(conditions):
        means, offsets = [], []
        for t_i, target_name in enumerate(targets_order):
            row = decoding_df[(decoding_df["target"] == target_name) &
                               (decoding_df["condition"] == condition)].iloc[0]
            means.append(row["mean_r2"])
            offset = x[t_i] + (c_i - 0.5) * bar_width
            offsets.append(offset)
            fold_vals = row["fold_r2"]
            ax.scatter([offset] * len(fold_vals), fold_vals, color="black",
                       s=18, zorder=3, alpha=0.7)
        ax.bar(offsets, means, width=bar_width, color=colors[condition],
               alpha=0.85, label=condition, zorder=2)

    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(targets_order, fontsize=11)
    ax.set_ylabel("Held-out R²", fontsize=11)
    ax.set_title("z_phy decoding of shadow nutrients: real vs shuffled control", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_z_phy_predicted_vs_true(prediction_results, save_path=None):
    """
    Scatter plot(s) of predicted vs true values for each target, with an
    identity (y=x) reference line and R² annotated in each subplot title.

    Parameters
    ----------
    prediction_results : dict[target_name] -> {"predictions": (N,) ndarray,
                          "true": (N,) ndarray}. Accepts either the dict
                          returned directly by decode_shadow_from_z_phy
                          (which has extra "mean_r2"/"fold_r2" keys per
                          target — these are ignored here), or a smaller
                          dict with just "predictions"/"true" per target
                          (e.g. one condition's entry pulled out of
                          compare_real_vs_shuffled_z_phy_decoding's second
                          return value).
    save_path           : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    target_names = list(prediction_results.keys())
    fig, axes = plt.subplots(1, len(target_names),
                              figsize=(5.5 * len(target_names), 5), squeeze=False)
    axes = axes[0]

    for ax, target_name in zip(axes, target_names):
        pred = np.asarray(prediction_results[target_name]["predictions"])
        true = np.asarray(prediction_results[target_name]["true"])

        valid = ~np.isnan(pred)
        pred, true = pred[valid], true[valid]
        r2 = r2_score(true, pred) if len(true) > 1 else float("nan")

        ax.scatter(true, pred, s=10, alpha=0.4, color="steelblue", edgecolors="none")
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], color="crimson", ls="--", lw=1.2, label="y = x")

        ax.set_xlabel(f"True {target_name}", fontsize=10)
        ax.set_ylabel(f"Predicted {target_name}", fontsize=10)
        ax.set_title(f"{target_name}: predicted vs true  (R² = {r2:.3f})", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_top_z_phy_dimensions_against_shadow(z_phy, targets, corr_df, episode_idx=None,
                                                top_k=3, episode_to_plot=0, save_path=None):
    """
    Plot the top-k z_phy dimensions (by absolute correlation with the
    shadow targets, from corr_df) over time for ONE episode, with hunger
    and fullness overlaid (min-max normalised to [0, 1] for visual
    comparability against z_phy's own scale, which is unconstrained).

    This is a qualitative/illustrative plot. A dimension shown here having
    a visually similar trace to hunger or fullness is consistent with that
    dimension being hunger/fullness-associated; it does not by itself
    establish that the dimension is dedicated to encoding it (see this
    module's section header on language) — the decoding tests are the
    quantitative evidence.

    Parameters
    ----------
    z_phy           : (N, z_dim) ndarray — pooled across episodes (as from
                       capture_phy_bottleneck_shadow_data), OR a single
                       episode's (T, z_dim) array. If pooled, episode_idx
                       must be provided to select episode_to_plot's rows;
                       if a single episode's array, leave episode_idx=None.
    targets         : dict[str, ndarray] — same row alignment as z_phy.
    corr_df         : pandas.DataFrame — output of
                       compute_z_phy_shadow_correlations, used only to rank
                       dimensions by abs_r (top_k highest, across all
                       targets combined).
    episode_idx     : optional (N,) int array — required if z_phy/targets
                       are pooled across multiple episodes; selects the rows
                       belonging to episode_to_plot. Leave None if z_phy/
                       targets already represent a single episode.
    top_k           : int — number of top-correlated z_phy dimensions to plot.
    episode_to_plot : int — which episode_idx value to select (ignored if
                       episode_idx is None).
    save_path       : optional path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if episode_idx is not None:
        mask = episode_idx == episode_to_plot
        if not mask.any():
            raise ValueError(f"episode_to_plot={episode_to_plot} not found in episode_idx.")
        z_seg = z_phy[mask]
        targets_seg = {name: np.asarray(y)[mask] for name, y in targets.items()}
    else:
        z_seg = z_phy
        targets_seg = {name: np.asarray(y) for name, y in targets.items()}

    top_dims = (
        corr_df.groupby("latent_dim")["abs_r"].max()
        .sort_values(ascending=False)
        .head(top_k)
        .index.tolist()
    )

    t = np.arange(z_seg.shape[0])
    fig, ax = plt.subplots(figsize=(14, 5))

    def _norm01(arr):
        arr = np.asarray(arr, dtype=np.float64)
        lo, hi = arr.min(), arr.max()
        rng = (hi - lo) if (hi - lo) > 1e-8 else 1.0
        return (arr - lo) / rng

    dim_colors = plt.cm.tab10(np.linspace(0, 1, max(top_k, 1)))
    for k_i, dim in enumerate(top_dims):
        ax.plot(t, _norm01(z_seg[:, dim]), color=dim_colors[k_i], lw=1.4,
                label=f"z_phy_{dim} (norm.)")

    # Reference lines for every shadow target present (not just two named
    # ones) — cycles through a small set of dash patterns so an arbitrary
    # number of targets (hunger, fullness, cck, ghrelin, glp_1, pyy, ...)
    # stay visually distinguishable from each other and from the z_phy
    # lines above (all shadow lines share black so they read as a distinct
    # group from the colored z_phy dimensions).
    SHADOW_LINE_STYLES = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1)),
                            (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))]
    for t_i, name in enumerate(targets_seg.keys()):
        style = SHADOW_LINE_STYLES[t_i % len(SHADOW_LINE_STYLES)]
        ax.plot(t, _norm01(targets_seg[name]), color="black", ls=style,
                 lw=1.8, alpha=0.8, label=f"{name} (norm.)")

    ax.set_xlabel("Episode step", fontsize=11)
    ax.set_ylabel("Min-max normalised value [0, 1]", fontsize=11)
    shadow_label = "/".join(targets_seg.keys())
    ax.set_title(
        f"Top {top_k} {shadow_label}-associated z_phy dimensions vs shadow "
        f"traces (episode {episode_to_plot if episode_idx is not None else '—'})",
        fontsize=12,
    )
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Thin wrapper — orchestrates the functions above, no analysis logic of its
# own. Kept separate so each step above remains independently callable/
# testable; this just saves typing the standard sequence out by hand.
# ──────────────────────────────────────────────────────────────────────────────

def run_z_phy_shadow_analysis(agent, env, n_episodes=20, deterministic=False,
                                method="pearson", alpha=1.0, n_splits=5, seed=0,
                                top_k=3, episode_to_plot=0, network="actor",
                                output_dir=None, run_tsne_step=True,
                                tsne_perplexity=30.0, tsne_max_samples=5000,
                                shadow_names=None):
    """
    Run the full z_phy <-> hunger/fullness analysis pipeline end to end:
    capture -> correlate (+ MSE) -> decode (real vs shuffled) -> t-SNE
    (descriptive) -> plot.

    This is a convenience wrapper, not where the logic lives — see the
    individual functions above (capture_phy_bottleneck_shadow_data,
    compute_z_phy_shadow_correlations, compare_real_vs_shuffled_z_phy_decoding,
    run_tsne, tsne_shadow_correlation_and_mse, plot_z_phy_shadow_correlation_heatmap,
    plot_z_phy_shadow_mse_heatmap, plot_z_phy_decoding_comparison,
    plot_z_phy_predicted_vs_true, plot_top_z_phy_dimensions_against_shadow,
    plot_tsne_scatter) for what each step actually does and how to call it
    independently.

    Parameters
    ----------
    agent, env      : see capture_phy_bottleneck_shadow_data.
    n_episodes      : episodes to pool for capture (also reused as the
                       episode pool for the single-episode qualitative plot
                       — episode_to_plot indexes into this same pool).
    deterministic   : passed through to capture.
    method          : "pearson" or "spearman", passed to
                       compute_z_phy_shadow_correlations and
                       tsne_shadow_correlation_and_mse.
    alpha, n_splits : passed to the decoding functions.
    seed            : passed to shuffle_z_within_episodes AND to run_tsne's
                       random_state (same seed reused for both — fine since
                       they affect unrelated, independent operations).
    top_k           : passed to plot_top_z_phy_dimensions_against_shadow.
    episode_to_plot : which pooled episode index to use for the qualitative
                       time-series plot.
    network         : "actor" or "critic", passed to capture.
    output_dir      : if provided, every plot is saved under this directory
                       with a fixed filename; directory is created if needed.
                       If None, plots are produced but not saved to disk.
    run_tsne_step   : if True (default), additionally fits a 2D t-SNE
                       embedding of the pooled z_phy and reports its
                       (descriptive-only — see run_tsne's docstring)
                       correlation/MSE against hunger/fullness, plus a
                       coloured scatter plot. Set False to skip this and
                       only run the original correlation/decoding pipeline.
    tsne_perplexity : passed to run_tsne.
    tsne_max_samples : passed to run_tsne as `max_samples` — caps the
                       number of pooled rows fitted in the t-SNE step (with
                       deterministic subsampling above this cap). This
                       guards against excessive memory/time use — and a
                       silently crashed kernel — when n_episodes x episode
                       length produces a very large pool (e.g. long,
                       multi-cycle FoodEnv episodes). See run_tsne's
                       docstring for details. Lower it if you still see
                       memory pressure; raise it (or pass None) only if
                       you have headroom to spare.
    shadow_names     : list of shadow-nutrient names to analyze, or None
                       (default) to use every nutrient currently configured
                       in env_config.SHADOW_NUTRIENT_CONFIG (i.e.
                       env.shadow_nutrient_names). Passed straight to
                       capture_phy_bottleneck_shadow_data — see that
                       function if you want to restrict to a subset (e.g.
                       just the original ["hunger", "fullness"]) instead of
                       everything the env has loaded.

    Returns
    -------
    dict with keys:
        "z_phy", "targets", "episode_idx", "meta"   — from capture
        "corr_df"                                    — from compute_z_phy_shadow_correlations
                                                         (now includes an 'mse' column)
        "decoding_comparison_df", "decoding_predictions" — from compare_real_vs_shuffled_z_phy_decoding
        "tsne_coords"                                 — (min(N, tsne_max_samples), 2) ndarray,
                                                         or None if run_tsne_step=False. Rows
                                                         correspond to a subsample of z_phy when
                                                         N > tsne_max_samples — see run_tsne.
        "tsne_corr_mse_df"                            — from tsne_shadow_correlation_and_mse, or None
        "figures"                                     — dict of the matplotlib Figures produced
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    def _path(name):
        return os.path.join(output_dir, name) if output_dir is not None else None

    z_phy, targets, episode_idx, meta = capture_phy_bottleneck_shadow_data(
        agent, env, n_episodes=n_episodes, deterministic=deterministic,
        network=network, shadow_names=shadow_names,
    )

    corr_df = compute_z_phy_shadow_correlations(z_phy, targets, method=method)

    decoding_comparison_df, decoding_predictions = compare_real_vs_shuffled_z_phy_decoding(
        z_phy, targets, episode_idx, alpha=alpha, n_splits=n_splits, seed=seed,
    )

    figures = {}
    figures["correlation_heatmap"] = plot_z_phy_shadow_correlation_heatmap(
        corr_df, save_path=_path("z_phy_shadow_correlation_heatmap.png"),
    )
    figures["mse_heatmap"] = plot_z_phy_shadow_mse_heatmap(
        corr_df, save_path=_path("z_phy_shadow_mse_heatmap.png"),
    )
    figures["decoding_comparison"] = plot_z_phy_decoding_comparison(
        decoding_comparison_df, save_path=_path("z_phy_decoding_comparison.png"),
    )

    real_only_predictions = {
        target_name: decoding_predictions[(target_name, "real_z_phy")]
        for target_name in targets
    }
    figures["predicted_vs_true"] = plot_z_phy_predicted_vs_true(
        real_only_predictions, save_path=_path("z_phy_predicted_vs_true.png"),
    )

    figures["top_dimensions_vs_shadow"] = plot_top_z_phy_dimensions_against_shadow(
        z_phy, targets, corr_df, episode_idx=episode_idx, top_k=top_k,
        episode_to_plot=episode_to_plot,
        save_path=_path("z_phy_top_dimensions_vs_shadow.png"),
    )

    tsne_coords, tsne_corr_mse_df = None, None
    if run_tsne_step:
        print("\n── t-SNE on pooled z_phy (descriptive only — see run_tsne "
              "docstring) ──")
        tsne_coords, _tsne_obj, tsne_sample_idx = run_tsne(
            z_phy, n_components=2, perplexity=tsne_perplexity, seed=seed,
            max_samples=tsne_max_samples,
        )
        # If run_tsne subsampled (large pool), targets must be subsampled
        # to the SAME rows so shapes line up with tsne_coords.
        tsne_targets = {name: np.asarray(y)[tsne_sample_idx]
                          for name, y in targets.items()}
        tsne_corr_mse_df = tsne_shadow_correlation_and_mse(
            tsne_coords, tsne_targets, method=method,
        )
        figures["tsne_scatter"] = plot_tsne_scatter(
            tsne_coords, tsne_targets, save_path=_path("z_phy_tsne_scatter.png"),
        )

    return {
        "z_phy": z_phy,
        "targets": targets,
        "episode_idx": episode_idx,
        "meta": meta,
        "corr_df": corr_df,
        "decoding_comparison_df": decoding_comparison_df,
        "decoding_predictions": decoding_predictions,
        "tsne_coords": tsne_coords,
        "tsne_corr_mse_df": tsne_corr_mse_df,
        "figures": figures,
    }