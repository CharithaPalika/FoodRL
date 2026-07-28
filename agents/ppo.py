import os
import json
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import warnings
from collections import deque
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.model import (
    SharedActorCritic, Actor, Critic,
    sample_continuous_action, recompute_log_prob,
    SharedDiscreteActorCritic, DiscreteActor,
    sample_discrete_action, recompute_discrete_log_prob,
)
from utils.utils import obs_to_tensors
from envs.env import FoodEnv
from envs.env_config import DISCRETE_EAT_AMOUNT


def _amplitude_to_jsonable(value):
    """
    Normalize an amplitude value (scalar or vector-like — python
    float/int, list, numpy array, torch tensor) into something
    json.dump can serialize directly for checkpoint_meta: a plain float
    for scalars, or a plain list of floats for vectors. Used wherever
    self.args['amplitude' / 'phy_amplitude' / 'food_amplitude'] is
    written, so checkpoint_meta.json never fails to serialize just
    because a numpy array or tensor was passed through.
    """
    if isinstance(value, (int, float)):
        return float(value)
    arr = np.asarray(
        value.detach().cpu().numpy() if torch.is_tensor(value) else value,
        dtype=float,
    ).reshape(-1)
    return arr.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# PPO Agent  —  Continuous OR discrete action space, with sleep/wake cycle
#               support. Mode is read from env.is_continuous (set on FoodEnv).
#
# Continuous (env.is_continuous=True):
#   Action:  amounts ∈ [0, 1]^num_foods  (one consumption fraction per slot)
#   Policy:  Clipped Gaussian — network predicts (mu, log_std) per food slot.
#            Samples x_raw ~ N(mu, std), clips to [0,1] for the env.
#            Log-prob is computed on x_raw (unclipped) — standard PPO practice.
#
# Discrete (env.is_continuous=False):
#   Action:  a single Categorical choice in {0, ..., num_foods}.
#            0 = eat nothing; k>=1 = eat menu slot (k-1) at a fixed amount
#            (env_config.DISCRETE_EAT_AMOUNT). At most one slot per step.
#   Policy:  Categorical(logits=...) over num_foods + 1 outcomes.
#
# In both modes, during sleep the env ignores the agent's output — the agent
# still produces an action so the policy gradient flows, but no absorption
# occurs.
#
# Observation: [nutrient_0…N-1, is_awake, time_in_cycle]
#          state_size is read directly from env.observation_space so the agent
#          adapts automatically to however many nutrients are active + the +2
#          cycle dims added by the env.
# ──────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, env, device="cpu", **kwargs):

        defaults = dict(
            gamma=0.99,
            lam=0.95,
            limit_delta=1.0,
            clip_eps=0.2,
            value_coeff=0.5,
            entropy_coeff=0.01,
            max_grad_norm=0.5,
            shared=False,
            shared_ac_network=None,
            actor_network=None,
            critic_network=None,
            seed=None,
            # Trunk / head architecture — ignored if a network object is
            # provided directly via shared_ac_network / actor_network /
            # critic_network.
            hidden=256,
            # ── Unified bottleneck configuration ───────────────────────────
            # A single nested config replaces the old scattered
            # bottleneck_dim / separate_branches / *_head_bottleneck_dim args.
            # Shape depends on `shared`:
            #
            #   shared=True :
            #       {"trunk": <trunk_spec>,
            #        "actor_head":  <head_spec>,
            #        "critic_head": <head_spec>}
            #
            #   shared=False:
            #       {"actor":  {"trunk": <trunk_spec>, "head": <head_spec>},
            #        "critic": {"trunk": <trunk_spec>, "head": <head_spec>}}
            #
            #   <trunk_spec> : None                        → no bottleneck
            #                  int                         → combined bottleneck
            #                  {"phy": int|None,
            #                   "food": int|None}          → separate branches
            #   <head_spec>  : None                        → no bottleneck
            #                  int                         → bottleneck dim
            #
            # Any key omitted or None means "no bottleneck there". Passing
            # None (the default) means no bottlenecks anywhere — plain MLP
            # trunks, heads read the trunk output directly. Actor and critic
            # are configured INDEPENDENTLY in non-shared mode, so their trunk
            # bottlenecks can differ (impossible in the old design without
            # hand-building the networks).
            bottlenecks=None,
        )

        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown init args: {unknown}")

        self.args   = {**defaults, **kwargs}
        self.device = device
        self.env    = env
        # Records the last amplitude / bias set per bottleneck name
        # (JSON-serializable), for checkpoint_meta visibility only — both are
        # runtime-only and are NOT restored on load (see set_amplitude/set_bias).
        self._amplitude_record = {}
        self._bias_record = {}

        # Action-space mode is owned by the env, not duplicated as agent
        # config — avoids the two ever disagreeing about action format.
        self.is_continuous = bool(getattr(env, "is_continuous", True))

        # ── Architectural checks ───────────────────────────────────────────
        if self.args["shared"]:
            if (self.args["actor_network"] is not None
                    or self.args["critic_network"] is not None):
                raise ValueError(
                    "shared=True but actor_network / critic_network provided; "
                    "use shared_ac_network instead."
                )
        else:
            if self.args["shared_ac_network"] is not None:
                raise ValueError(
                    "shared=False but shared_ac_network provided; "
                    "use actor_network / critic_network instead."
                )

        # ── Environment specs ──────────────────────────────────────────────
        # state_shape now includes the +2 cycle dims (is_awake, time_in_cycle)
        # automatically because the env sets state_dim = num_nutrients + 2.
        try:
            num_foods      = env.num_foods
            state_shape    = env.observation_space["physiological_state"].shape[0]
            food_emb_shape = env.observation_space["food_embeddings"].shape
            food_flat_size = food_emb_shape[0] * food_emb_shape[1]
        except Exception as e:
            raise ValueError(
                "Environment observation space does not match expected format."
            ) from e

        # ── Build networks ─────────────────────────────────────────────────
        seed   = self.args["seed"]
        hidden = self.args["hidden"]
        # Normalise the bottleneck config into the canonical nested form and
        # stash it (used for network construction, checkpoint_meta, and
        # from_checkpoint reconstruction).
        self.bottlenecks_config = self._normalize_bottlenecks_config(
            self.args["bottlenecks"], self.args["shared"]
        )
        cfg = self.bottlenecks_config

        if self.args["shared"]:
            if self.args["shared_ac_network"] is not None:
                self.policy = self.args["shared_ac_network"]
            else:
                cls = SharedActorCritic if self.is_continuous else SharedDiscreteActorCritic
                self.policy = cls(
                    state_size=state_shape,
                    food_flat_size=food_flat_size,
                    num_foods=num_foods,
                    hidden=hidden,
                    trunk=cfg.get("trunk"),
                    actor_head=cfg.get("actor_head"),
                    critic_head=cfg.get("critic_head"),
                    seed=seed,
                )
            self.policy = self.policy.to(self.device)
            self.actor  = None
            self.critic = None
        else:
            actor_cfg  = cfg.get("actor", {})
            critic_cfg = cfg.get("critic", {})
            if self.args["actor_network"] is not None:
                self.actor = self.args["actor_network"]
            else:
                cls = Actor if self.is_continuous else DiscreteActor
                self.actor = cls(
                    state_size=state_shape,
                    food_flat_size=food_flat_size,
                    num_foods=num_foods,
                    hidden=hidden,
                    trunk=actor_cfg.get("trunk"),
                    head=actor_cfg.get("head"),
                    seed=seed,
                )
            self.actor = self.actor.to(self.device)
            self.critic = (
                self.args["critic_network"]
                if self.args["critic_network"] is not None
                else Critic(
                    state_size=state_shape,
                    food_flat_size=food_flat_size,
                    hidden=hidden,
                    trunk=critic_cfg.get("trunk"),
                    head=critic_cfg.get("head"),
                    seed=seed,
                )
            ).to(self.device)
            self.policy = None

        # ── Global bottleneck registry ─────────────────────────────────────
        # Maps a unique, human-addressable name to each bottleneck module that
        # actually exists, so any one can be probed by name (set_amplitude).
        # Shared: names come straight from the policy (trunk / trunk_phy /
        # trunk_food / actor_head / critic_head). Non-shared: the actor's and
        # critic's local names are prefixed so they never collide
        # (actor_trunk, actor_head, critic_trunk, critic_head, …).
        self.bottlenecks = {}
        if self.args["shared"]:
            self.bottlenecks.update(self.policy.bottlenecks)
        else:
            for local, mod in self.actor.bottlenecks.items():
                self.bottlenecks[f"actor_{local}"] = mod
            for local, mod in self.critic.bottlenecks.items():
                self.bottlenecks[f"critic_{local}"] = mod

        self.policy_optimizer = None
        self.actor_optimizer  = None
        self.critic_optimizer = None

        # LR schedulers (one per active optimizer). Stay None unless train() is
        # called with lr_schedule != None — i.e. constant LR by default.
        self.policy_scheduler = None
        self.actor_scheduler  = None
        self.critic_scheduler = None
        self._lr_gamma        = None

        # Best-checkpoint tracking (best single-episode return seen so far).
        self._best_return  = -float("inf")
        self._best_episode = None

        # ── Per-episode training log (one value per completed episode) ─────
        # These are the arrays saved to training_log.npz for later plotting.
        self.episode_returns     = []   # total reward per episode
        self.episode_distance    = []   # total L2-to-target distance per ep
                                        #   (summed over every step)
        self.episode_food_count  = []   # total number of food ITEMS eaten per
                                        #   ep (count of eating events, not
                                        #   fractional amounts)
        self.episode_actor_loss  = []   # actor loss per ep (latest PPO update
                                        #   mean at the time the ep finished)
        self.episode_critic_loss = []   # critic loss per ep (same convention)
        # Kept for backward-compatible console printing (fractional amounts).
        self.episode_consumption = []
        # Latest PPO-update mean losses/entropy, snapshotted onto each episode
        # as it completes (NaN until the first update runs).
        self._last_actor_loss  = float("nan")
        self._last_critic_loss = float("nan")
        self._last_entropy     = float("nan")

        # ── TD-error logging (opt-in; OFF by default so training is unchanged) ──
        # When enabled via enable_td_logging(), _compute_gae records the per-step
        # TD error BEFORE and AFTER the one-sided limit_delta clip, one array per
        # rollout, and can print a raw-vs-clipped summary. Read with get_td_log().
        self._td_log_enabled = False
        self._td_print_every = None
        self._td_rollout_i   = 0
        self._td_raw_log     = []   # list of per-rollout raw (unclipped) delta arrays
        self._td_clipped_log = []   # list of per-rollout clipped delta arrays

    def enable_td_logging(self, enabled=True, reset=True, print_every=None):
        """
        Record (and optionally print) per-rollout TD errors straight from
        _compute_gae, BEFORE and AFTER the one-sided `limit_delta` clip. Off by
        default, so normal training is unaffected.

        enabled     : turn logging on/off.
        reset       : clear any previously logged rollouts (and the counter).
        print_every : if set to N, print a raw-vs-clipped summary line every N
                      rollouts (None = log silently). The first rollout always
                      prints when print_every is set.
        """
        self._td_log_enabled = bool(enabled)
        self._td_print_every = print_every
        # Only clear when (re)starting logging — disabling must NOT wipe the log,
        # otherwise enable_td_logging(False) after training loses everything.
        if reset and enabled:
            self._td_rollout_i   = 0
            self._td_raw_log     = []
            self._td_clipped_log = []

    def get_td_log(self):
        """
        Return (raw_deltas, clipped_deltas): each a list of per-rollout 1-D numpy
        arrays, index-aligned. Concatenate for the pooled TD-error distribution;
        the difference between the two is exactly what the clip removed.
        """
        return self._td_raw_log, self._td_clipped_log

    def training_log(self):
        """Return the per-episode training log as a dict of equal-length
        float arrays: reward, distance, consumption (food-item count),
        actor_loss, critic_loss. This is exactly what save_training_log /
        the checkpoint bundles write to *_log.npz."""
        return {
            "reward":      np.asarray(self.episode_returns,     dtype=np.float64),
            "distance":    np.asarray(self.episode_distance,    dtype=np.float64),
            "consumption": np.asarray(self.episode_food_count,  dtype=np.float64),
            "actor_loss":  np.asarray(self.episode_actor_loss,  dtype=np.float64),
            "critic_loss": np.asarray(self.episode_critic_loss, dtype=np.float64),
        }

    def save_training_log(self, path):
        """Write the per-episode training log to `path` (an .npz). Keys:
        reward, distance, consumption, actor_loss, critic_loss."""
        np.savez(path, **self.training_log())
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Bottleneck config normalisation
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_bottlenecks_config(bottlenecks, shared):
        """
        Validate + canonicalise the `bottlenecks` config into a plain dict:

          shared=True  → {"trunk": <spec>, "actor_head": <spec>,
                          "critic_head": <spec>}   (missing keys → None)
          shared=False → {"actor":  {"trunk": <spec>, "head": <spec>},
                          "critic": {"trunk": <spec>, "head": <spec>}}

        None anywhere means "no bottleneck there". Raises a clear error if the
        config shape does not match `shared` (e.g. an "actor"/"critic" key in
        shared mode, or a "trunk" top-level key in non-shared mode).
        """
        bottlenecks = {} if bottlenecks is None else dict(bottlenecks)

        if shared:
            allowed = {"trunk", "actor_head", "critic_head"}
            extra = set(bottlenecks) - allowed
            if extra:
                raise ValueError(
                    f"shared=True bottlenecks config accepts only {sorted(allowed)}, "
                    f"got unexpected {sorted(extra)}. (In non-shared mode use "
                    "{'actor': {...}, 'critic': {...}}.)"
                )
            return {
                "trunk":       bottlenecks.get("trunk"),
                "actor_head":  bottlenecks.get("actor_head"),
                "critic_head": bottlenecks.get("critic_head"),
            }

        # non-shared
        allowed = {"actor", "critic"}
        extra = set(bottlenecks) - allowed
        if extra:
            raise ValueError(
                f"shared=False bottlenecks config accepts only {sorted(allowed)}, "
                f"got unexpected {sorted(extra)}. Each maps to "
                "{'trunk': <spec>, 'head': <spec>}. (In shared mode use "
                "top-level 'trunk'/'actor_head'/'critic_head' keys instead.)"
            )
        out = {}
        for role in ("actor", "critic"):
            sub = bottlenecks.get(role) or {}
            sub_extra = set(sub) - {"trunk", "head"}
            if sub_extra:
                raise ValueError(
                    f"bottlenecks['{role}'] accepts only 'trunk' and 'head', "
                    f"got unexpected {sorted(sub_extra)}."
                )
            out[role] = {"trunk": sub.get("trunk"), "head": sub.get("head")}
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Amplitude control  (name-addressed — see self.bottlenecks / list_bottlenecks)
    # ──────────────────────────────────────────────────────────────────────────

    def list_bottlenecks(self):
        """
        Return {name → {"dim", "amplitude", "bias"}} for every bottleneck in the
        network, in a stable order. `name` is what set_amplitude / set_bias
        expect; `dim` is the bottleneck width; `amplitude` is the current
        per-neuron multiplicative gain and `bias` the additive offset (each a
        float if uniform, else a list). Use this to discover exactly which
        bottlenecks exist for the current architecture.
        """
        def _fmt(arr):
            uniform = bool(np.all(arr == arr.flat[0]))
            return float(arr.flat[0]) if uniform else arr.tolist()

        out = {}
        for name, mod in self.bottlenecks.items():
            out[name] = {
                "dim":       int(mod.z_dim),
                "amplitude": _fmt(mod.get_amplitude()),
                "bias":      _fmt(mod.get_bias()),
            }
        return out

    def set_amplitude(self, name_or_map, value=None):
        """
        Retune one or more bottlenecks' per-neuron gain live, addressed BY
        NAME — no rebuild of agent or env needed. Names come from
        self.bottlenecks / list_bottlenecks(), e.g. (non-shared) "actor_trunk",
        "actor_trunk_phy", "actor_head", "critic_trunk", "critic_head"; or
        (shared) "trunk", "trunk_phy", "trunk_food", "actor_head",
        "critic_head". Only names that actually exist for the current
        architecture are valid.

        Two call styles:
            agent.set_amplitude("actor_trunk", 2.5)          # one bottleneck
            agent.set_amplitude({"actor_trunk_phy": [0.0, 1.0, 1.0, ...],
                                 "critic_head": 0.0})        # several at once

        value : scalar — broadcast the same gain to every neuron of that
                bottleneck; OR a 1D sequence/array/tensor of length == that
                bottleneck's dim — one independent gain per neuron (e.g. zero a
                single neuron, leave the rest at 1.0). A wrong-length vector
                raises, naming the expected length.

        Persistence: the gain lives in a runtime-only buffer (persistent=False)
        — NOT saved in the .pt files and NOT restored by load_model /
        from_checkpoint. checkpoint_meta records the last-set values under
        "amplitudes" for reference only; re-call this after loading to
        reinstate a probing configuration. set_amplitude(name, 1.0) resets one
        bottleneck; reset_amplitudes() resets all.
        """
        if isinstance(name_or_map, dict):
            if value is not None:
                raise ValueError("pass a dict OR (name, value), not both.")
            items = list(name_or_map.items())
        else:
            if value is None:
                raise ValueError("set_amplitude(name, value) requires a value.")
            items = [(name_or_map, value)]

        for name, val in items:
            if name not in self.bottlenecks:
                raise KeyError(
                    f"no bottleneck named {name!r}. Available for this "
                    f"architecture: {list(self.bottlenecks.keys())}. "
                    "(Call agent.list_bottlenecks() to see dims too.)"
                )
            self.bottlenecks[name].set_amplitude(val)
            self._amplitude_record[name] = _amplitude_to_jsonable(val)

    def reset_amplitudes(self):
        """Reset every bottleneck's gain to a uniform 1.0 (no-op probe)."""
        for name in self.bottlenecks:
            self.bottlenecks[name].set_amplitude(1.0)
        self._amplitude_record = {}

    def set_bias(self, name_or_map, value=None):
        """
        Add a constant per-neuron OFFSET to one or more bottlenecks, addressed
        BY NAME. This is the additive counterpart of set_amplitude: the
        bottleneck activation becomes

            z = encoder(x) * amplitude + bias

        so `bias` shifts the latent code while `amplitude` scales it. Both are
        independent — a bias probe with amplitude left at 1.0 changes only the
        offset. Names come from self.bottlenecks / list_bottlenecks().

        Two call styles (same as set_amplitude):
            agent.set_bias("trunk_phy", 0.5)
            agent.set_bias({"trunk_phy": [0.5, 0.0, ...], "critic_head": -0.2})

        value : scalar — same offset added to every neuron of that bottleneck;
                OR a 1D sequence/array/tensor of length == that bottleneck's dim
                for an independent offset per neuron. Wrong-length vectors raise.

        Persistence: like amplitude, the offset lives in a runtime-only buffer
        (persistent=False) — NOT saved in the .pt files and NOT restored by
        load_model / from_checkpoint. checkpoint_meta records the last-set values
        under "biases" for reference only. set_bias(name, 0.0) resets one
        bottleneck; reset_biases() resets all.
        """
        if isinstance(name_or_map, dict):
            if value is not None:
                raise ValueError("pass a dict OR (name, value), not both.")
            items = list(name_or_map.items())
        else:
            if value is None:
                raise ValueError("set_bias(name, value) requires a value.")
            items = [(name_or_map, value)]

        for name, val in items:
            if name not in self.bottlenecks:
                raise KeyError(
                    f"no bottleneck named {name!r}. Available for this "
                    f"architecture: {list(self.bottlenecks.keys())}. "
                    "(Call agent.list_bottlenecks() to see dims too.)"
                )
            self.bottlenecks[name].set_bias(val)
            self._bias_record[name] = _amplitude_to_jsonable(val)

    def reset_biases(self):
        """Reset every bottleneck's additive offset to a uniform 0.0 (no-op)."""
        for name in self.bottlenecks:
            self.bottlenecks[name].set_bias(0.0)
        self._bias_record = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _init_optimizers(self, actor_lr, critic_lr, shared_ac_lr):
        if self.args["shared"]:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=shared_ac_lr
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=critic_lr
            )

    def _init_schedulers(self, lr_schedule, num_episodes):
        """
        Build a per-optimizer LR scheduler, or leave them None (constant LR).

        lr_schedule=None  → scheduling OFF: LR stays constant for the whole run
                            (the original behaviour). This is the on/off switch.

        lr_schedule=dict  → scheduling ON. Recognised keys:
            "type"       : only "exponential" is supported. LR is multiplied by
                           a fixed gamma once PER COMPLETED EPISODE, so after E
                           episodes  lr = lr0 * gamma**E  (geometric decay).
            "final_frac" : target fraction of the INITIAL LR to reach by the
                           final episode (default 0.1 → decays to 10%). gamma is
                           derived so gamma**num_episodes == final_frac, i.e.
                           gamma = final_frac ** (1 / num_episodes). Each
                           optimizer decays from its OWN base LR.
            "gamma"      : OPTIONAL explicit per-episode multiplier. If given,
                           it overrides final_frac (num_episodes is then
                           irrelevant to the rate).

        Applies to whichever optimizers exist: the shared optimizer, or the
        separate actor + critic optimizers.
        """
        self.policy_scheduler = None
        self.actor_scheduler  = None
        self.critic_scheduler = None
        self._lr_gamma        = None
        if lr_schedule is None:
            return

        cfg        = dict(lr_schedule)
        sched_type = cfg.get("type", "exponential")
        if sched_type != "exponential":
            raise ValueError(
                f"lr_schedule['type']={sched_type!r} not supported — only "
                "'exponential' is implemented."
            )

        if cfg.get("gamma") is not None:
            gamma = float(cfg["gamma"])
            if not (0.0 < gamma <= 1.0):
                raise ValueError(f"lr_schedule gamma must be in (0, 1], got {gamma}.")
        else:
            final_frac = float(cfg.get("final_frac", 0.1))
            if not (0.0 < final_frac <= 1.0):
                raise ValueError(
                    f"lr_schedule final_frac must be in (0, 1], got {final_frac}."
                )
            # gamma**num_episodes == final_frac  →  reach final_frac × LR by the
            # last episode (per-episode stepping; see train()).
            gamma = final_frac ** (1.0 / max(int(num_episodes), 1))

        self._lr_gamma = gamma
        Sched = torch.optim.lr_scheduler.ExponentialLR
        if self.args["shared"]:
            self.policy_scheduler = Sched(self.policy_optimizer, gamma=gamma)
        else:
            self.actor_scheduler  = Sched(self.actor_optimizer,  gamma=gamma)
            self.critic_scheduler = Sched(self.critic_optimizer, gamma=gamma)

    def _step_schedulers(self):
        """Advance every active LR scheduler by one episode (no-op if off)."""
        for sched in (self.policy_scheduler, self.actor_scheduler, self.critic_scheduler):
            if sched is not None:
                sched.step()

    def _current_lrs(self):
        """Current LR of each active optimizer, keyed by role (for logging)."""
        lrs = {}
        if self.policy_optimizer is not None:
            lrs["shared_ac_lr"] = self.policy_optimizer.param_groups[0]["lr"]
        if self.actor_optimizer is not None:
            lrs["actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]
        if self.critic_optimizer is not None:
            lrs["critic_lr"] = self.critic_optimizer.param_groups[0]["lr"]
        return lrs

    def _compute_gae(self, rewards, values, dones):
        """
        Generalised Advantage Estimation with optional ONE-SIDED TD-error
        clipping, controlled by `limit_delta`. The SIGN of limit_delta selects
        which side of the raw TD error is clipped:

            limit_delta is None  →  no clipping
            limit_delta  > 0     →  UPPER clip:  delta = min(delta, limit_delta)
                                    caps POSITIVE TD error (better-than-expected
                                    outcomes are damped; negative error is left
                                    untouched)
            limit_delta  < 0     →  LOWER clip:  delta = max(delta, limit_delta)
                                    floors NEGATIVE TD error (worse-than-expected
                                    outcomes are damped; positive error is left
                                    untouched)
            limit_delta == 0     →  no clipping (the side is ambiguous at zero,
                                    so this is treated as a neutral no-op rather
                                    than silently picking one)

        Note this is deliberately one-sided, not a symmetric +/- clamp: the two
        directions are studied separately (see td_clip_retrain.ipynb).
        """
        advs = np.zeros_like(rewards)
        gae  = 0.0
        c    = self.args["limit_delta"]

        # Optional TD-error logging (raw = before clip, clipped = after clip).
        log_td = getattr(self, "_td_log_enabled", False)
        if log_td:
            raw_deltas     = np.empty(len(rewards), dtype=np.float64)
            clipped_deltas = np.empty(len(rewards), dtype=np.float64)

        for t in reversed(range(len(rewards))):
            raw_delta = rewards[t] + self.args["gamma"] * values[t + 1] * (1 - dones[t]) - values[t]
            delta     = raw_delta
            if c is not None and c != 0:
                delta = min(delta, c) if c > 0 else max(delta, c)
            if log_td:
                raw_deltas[t]     = raw_delta
                clipped_deltas[t] = delta
            gae     = delta + self.args["gamma"] * self.args["lam"] * (1 - dones[t]) * gae
            advs[t] = gae

        if log_td:
            self._td_rollout_i += 1
            self._td_raw_log.append(raw_deltas)
            self._td_clipped_log.append(clipped_deltas)
            pe = self._td_print_every
            if pe and (self._td_rollout_i == 1 or self._td_rollout_i % pe == 0):
                n_clip = int(np.sum(clipped_deltas != raw_deltas))
                print(
                    f"[TD rollout {self._td_rollout_i:4d}] "
                    f"before clip: mean={raw_deltas.mean():+.5f} "
                    f"min={raw_deltas.min():+.5f} max={raw_deltas.max():+.5f}  |  "
                    f"after clip: mean={clipped_deltas.mean():+.5f} "
                    f"min={clipped_deltas.min():+.5f} max={clipped_deltas.max():+.5f}  |  "
                    f"{100.0 * n_clip / len(raw_deltas):.1f}% clipped (limit_delta={c})"
                )

        returns = advs + values[:-1]
        return advs, returns

    # ──────────────────────────────────────────────────────────────────────────
    # Act
    # ──────────────────────────────────────────────────────────────────────────

    def act(self, obs, deterministic=False):
        """
        Returns a 6-tuple. Slot MEANING depends on self.is_continuous:

        continuous:
            amounts   : (num_foods,) tensor  in [0, 1]   — clipped for env
            log_prob  : scalar tensor                    — sum log-prob over foods
            value     : scalar tensor
            mu        : (num_foods,) tensor              — Gaussian mean
            log_std   : (num_foods,) tensor              — Gaussian log std
            x_raw     : (num_foods,) tensor              — unclipped sample
                                                            (stored for PPO update)

        discrete:
            amounts   : scalar long tensor  in {0, ..., num_foods} — the env
                        action directly (env.step() expands it internally)
            log_prob  : scalar tensor
            value     : scalar tensor
            mu        : (num_foods + 1,) tensor — logits (repurposed slot)
            log_std   : None                    — unused in discrete mode
            x_raw     : scalar long tensor       — sampled action index
                                                    (same value as `amounts`,
                                                    kept for PPO update symmetry
                                                    with the continuous path)
        """
        phy, food_flat = obs_to_tensors(obs, self.device)

        if self.is_continuous:
            if self.args["shared"]:
                mu, log_std, value = self.policy(phy, food_flat)
            else:
                mu, log_std = self.actor(phy, food_flat)
                value       = self.critic(phy, food_flat)

            if deterministic:
                # Inference: use the Gaussian mean, clipped to valid range
                amounts  = torch.clamp(mu, 0.0, 1.0)
                log_prob = None
                x_raw    = mu
            else:
                amounts, log_prob, _, x_raw = sample_continuous_action(mu, log_std)

            return amounts, log_prob, value, mu, log_std, x_raw

        else:
            if self.args["shared"]:
                logits, value = self.policy(phy, food_flat)
            else:
                logits = self.actor(phy, food_flat)
                value  = self.critic(phy, food_flat)

            if deterministic:
                # Inference: take the argmax action, no sampling.
                action   = torch.argmax(logits, dim=-1)
                log_prob = None
            else:
                action, log_prob, _ = sample_discrete_action(logits)

            amounts = action
            x_raw   = action
            log_std = None
            return amounts, log_prob, value, logits, log_std, x_raw

    # ──────────────────────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, log_wandb=True, printing=True, **kwargs):

        train_defaults = dict(
            num_episodes=10_000,
            shared_ac_lr=1e-3,
            actor_lr=1e-3,
            critic_lr=1e-3,
            # ── LR scheduler (opt-in) ──────────────────────────────────────
            # None (default) → constant LR for the whole run (original
            # behaviour). Pass a dict to enable per-episode exponential decay,
            # e.g. {"type": "exponential", "final_frac": 0.1} decays each LR to
            # 10% of its initial value by the final episode. See
            # _init_schedulers() for the full key list ("gamma" to override).
            lr_schedule=None,
            rollout_steps=512,
            ppo_epochs=5,
            minibatch_size=64,
            log_every_episodes=1, #50
            rolling_window=50,
            # Periodic checkpointing during training — independent of the
            # notebook's end-of-training save cell, which still runs as
            # before. Set checkpoint_every=None (default) to disable.
            checkpoint_every=None,     # save every N episodes; None = off
            checkpoint_dir="checkpoints",
        )

        unknown = set(kwargs) - set(train_defaults)
        if unknown:
            raise ValueError(f"Unknown train args: {unknown}")
        train_args = {**train_defaults, **kwargs}

        if train_args["checkpoint_every"] is not None and train_args["checkpoint_every"] <= 0:
            raise ValueError(
                f"checkpoint_every must be a positive int or None, "
                f"got {train_args['checkpoint_every']!r}."
            )

        self._init_optimizers(
            actor_lr=train_args["actor_lr"],
            critic_lr=train_args["critic_lr"],
            shared_ac_lr=train_args["shared_ac_lr"],
        )
        # Optional per-episode LR decay (None → constant LR). Built AFTER the
        # optimizers so each scheduler wraps its own optimizer.
        self._init_schedulers(train_args["lr_schedule"], train_args["num_episodes"])

        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn("wandb requested but not installed.", RuntimeWarning)
            log_wandb = False

        episode_actor_losses  = []
        episode_critic_losses = []
        episode_entropies     = []
        episode_total_losses  = []

        obs, _         = self.env.reset()
        ep_return      = 0.0
        episode_count  = 0
        ep_consumption = 0.0   # sum of amounts consumed during awake steps only
        ep_distance    = 0.0   # sum of per-step L2-to-target distance over the ep

        while episode_count < train_args["num_episodes"]:

            phy_buf, food_flat_buf                        = [], []
            act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], []
            # x_raw_buf: unclipped Gaussian samples — needed for correct
            # importance ratio during the PPO update (must not re-sample)
            x_raw_buf = []

            # ── Rollout ───────────────────────────────────────────────────
            for _ in range(train_args["rollout_steps"]):
                phy, food_flat = obs_to_tensors(obs, self.device)

                amounts, logp, value, mu, log_std, x_raw = self.act(obs, deterministic=False)
                value_sq = value.squeeze(-1).squeeze()
                logp_sq  = logp.squeeze()

                if self.is_continuous:
                    amounts_sq = amounts.squeeze(0)    # (num_foods,)
                    x_raw_sq   = x_raw.squeeze(0)       # (num_foods,)
                    amounts_np = amounts_sq.detach().cpu().numpy()
                    env_action = amounts_np
                    step_consumption = float(amounts_sq.sum().item())
                else:
                    amounts_sq = amounts.squeeze(0)     # scalar action index
                    x_raw_sq   = x_raw.squeeze(0)        # scalar action index
                    env_action = int(amounts_sq.detach().cpu().item())
                    # Consumption is DISCRETE_EAT_AMOUNT if a slot was eaten,
                    # else 0 — there's no vector to sum here.
                    step_consumption = DISCRETE_EAT_AMOUNT if env_action > 0 else 0.0

                # The env zeroes out amounts during sleep internally, but we
                # still record what the agent OUTPUT so the policy gradient
                # flows correctly through sleep steps.
                # For consumption tracking we only count awake steps.
                # We check is_awake from the observation: physiological_state[-2]
                agent_is_awake = float(obs["physiological_state"][-2]) > 0.5
                if agent_is_awake:
                    ep_consumption += step_consumption

                next_obs, reward, terminated, done, info = self.env.step(env_action)

                # Accumulate total episode distance = per-step L2 distance of the
                # nutrient state to its target centre, summed over the episode
                # (env._phys_nutrients / _norm_targets are the nutrient dims only,
                # no cycle scalars).
                ep_distance += float(np.linalg.norm(
                    self.env._phys_nutrients - self.env._norm_targets, ord=2))

                phy_buf.append(phy.squeeze(0))
                food_flat_buf.append(food_flat.squeeze(0))
                act_buf.append(amounts_sq.detach())
                x_raw_buf.append(x_raw_sq.detach())
                rew_buf.append(reward)
                done_buf.append(done)
                logp_buf.append(logp_sq.detach())
                val_buf.append(value_sq.detach())

                ep_return += reward
                obs = next_obs

                if done:
                    # Total food items the agent CHOSE to eat this episode =
                    # number of real eating events logged by the env. This is
                    # wake-only (the env zeroes the action during sleep and only
                    # logs inside the awake phase) and excludes "eat nothing" /
                    # sub-threshold non-eats. Read BEFORE reset() clears the log.
                    ep_food_count = len(self.env._step_consumption)

                    self.episode_returns.append(ep_return)
                    self.episode_consumption.append(ep_consumption)
                    self.episode_distance.append(ep_distance)
                    self.episode_food_count.append(ep_food_count)
                    self.episode_actor_loss.append(self._last_actor_loss)
                    self.episode_critic_loss.append(self._last_critic_loss)
                    episode_count  += 1

                    # Per-episode LR decay (no-op if lr_schedule was None).
                    # Only changes the LR VALUE used by FUTURE optimizer.step()
                    # calls, so stepping here at episode completion is correct.
                    self._step_schedulers()

                    # ── Best-checkpoint tracking (best single-episode return) ──
                    # Whenever this episode beats the best seen so far, record
                    # it and — if checkpointing is enabled — (over)write the
                    # dedicated best_* bundle alongside the periodic ones.
                    is_new_best = ep_return > self._best_return
                    if is_new_best:
                        self._best_return  = float(ep_return)
                        self._best_episode = episode_count

                    ep_return       = 0.0
                    ep_consumption  = 0.0
                    ep_distance     = 0.0
                    obs, _ = self.env.reset()

                    if train_args["checkpoint_every"] is not None:
                        if is_new_best:
                            best_path = self.save_best(
                                train_args["checkpoint_dir"], log_wandb=log_wandb,
                            )
                            if printing:
                                print(f"  [best] new best return "
                                      f"{self._best_return:.2f} @ ep{episode_count} "
                                      f"→ {best_path}_*")
                        if episode_count % train_args["checkpoint_every"] == 0:
                            ckpt_path = self.save_checkpoint(
                                train_args["checkpoint_dir"], episode_count,
                                log_wandb=log_wandb,
                            )
                            if printing:
                                print(f"  [checkpoint] saved to {ckpt_path}_*")

                    if episode_count % train_args["log_every_episodes"] == 0:
                        rolling_avg  = np.mean(self.episode_returns[-train_args["rolling_window"]:])
                        last_return  = self.episode_returns[-1]
                        last_food    = self.episode_food_count[-1]
                        last_dist    = self.episode_distance[-1]

                        if printing:
                            print(
                                f"Episode {episode_count:5d} | "
                                f"Last Return: {last_return:8.2f} | "
                                f"Rolling Avg({train_args['rolling_window']}): {rolling_avg:8.2f} | "
                                f"Food items: {last_food:4d} | "
                                f"Total Distance: {last_dist:8.2f} | "
                                f"Actor L: {self._last_actor_loss:7.3f} | "
                                f"Critic L: {self._last_critic_loss:7.3f}"
                            )

                        if log_wandb:
                            # Log the SAME per-episode quantities saved to
                            # training_log.npz, aligned to the episode index:
                            #   reward     — total episode reward
                            #   distance   — total L2-to-target distance (summed)
                            #   consumption— total food ITEMS eaten (count)
                            #   actor/critic_loss — most recent PPO update's mean
                            #                       (NaN before the first update)
                            wandb.log(
                                {
                                    "train/reward":             self.episode_returns[-1],
                                    "train/reward_rolling_avg": float(rolling_avg),
                                    "train/distance":           self.episode_distance[-1],
                                    "train/consumption":        self.episode_food_count[-1],
                                    "train/actor_loss":         self.episode_actor_loss[-1],
                                    "train/critic_loss":        self.episode_critic_loss[-1],
                                    "train/entropy":            self._last_entropy,
                                    # Live LR(s) — constant unless lr_schedule is on.
                                    **{f"train/{k}": v for k, v in self._current_lrs().items()},
                                },
                                step=episode_count,
                            )

                    if episode_count >= train_args["num_episodes"]:
                        break

            # ── Bootstrap value ───────────────────────────────────────────
            with torch.no_grad():
                phy_b, food_flat_b = obs_to_tensors(obs, self.device)
                if self.args["shared"]:
                    policy_out = self.policy(phy_b, food_flat_b)
                    val_boot   = policy_out[-1]   # value is always the last output
                else:
                    val_boot = self.critic(phy_b, food_flat_b)
                val_buf.append(val_boot.squeeze(-1).squeeze())

            # ── GAE ───────────────────────────────────────────────────────
            advs, rets = self._compute_gae(
                np.array(rew_buf),
                torch.stack(val_buf).cpu().numpy(),
                np.array(done_buf, dtype=np.float32),
            )
            advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            rets = torch.tensor(rets, dtype=torch.float32, device=self.device)

            phy_batch       = torch.stack(phy_buf)
            food_flat_batch = torch.stack(food_flat_buf)
            act_batch       = torch.stack(act_buf)     # (T, num_foods) continuous, (T,) discrete
            x_raw_batch     = torch.stack(x_raw_buf)  # (T, num_foods) continuous, (T,) discrete
            logp_old        = torch.stack(logp_buf)    # (T,)

            # ── PPO update ────────────────────────────────────────────────
            # Remember where this rollout's minibatch losses start, so we can
            # snapshot THIS update's mean actor/critic loss onto subsequent
            # episodes (per-episode-aligned loss logging).
            _loss_start = len(episode_actor_losses)
            for _ in range(train_args["ppo_epochs"]):
                idx = torch.randperm(len(phy_batch))

                for start in range(0, len(phy_batch), train_args["minibatch_size"]):
                    mb = idx[start : start + train_args["minibatch_size"]]

                    phy_mb  = phy_batch[mb]
                    food_mb = food_flat_batch[mb]

                    if self.is_continuous:
                        if self.args["shared"]:
                            mu_mb, log_std_mb, val = self.policy(phy_mb, food_mb)
                        else:
                            mu_mb, log_std_mb = self.actor(phy_mb, food_mb)
                            val               = self.critic(phy_mb, food_mb)

                        # Recompute log_prob using STORED x_raw — must not
                        # re-sample, otherwise the importance ratio is
                        # meaningless.
                        logp, entropy = recompute_log_prob(mu_mb, log_std_mb, x_raw_batch[mb])
                    else:
                        if self.args["shared"]:
                            logits_mb, val = self.policy(phy_mb, food_mb)
                        else:
                            logits_mb = self.actor(phy_mb, food_mb)
                            val       = self.critic(phy_mb, food_mb)

                        # Recompute log_prob using STORED discrete action
                        # indices — must not re-sample, same reasoning as
                        # the continuous path above.
                        action_mb     = x_raw_batch[mb].long()
                        logp, entropy = recompute_discrete_log_prob(logits_mb, action_mb)

                    ratio  = torch.exp(logp - logp_old[mb])
                    surr1  = ratio * advs[mb]
                    surr2  = torch.clamp(ratio,
                                         1 - self.args["clip_eps"],
                                         1 + self.args["clip_eps"]) * advs[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss  = ((val.squeeze() - rets[mb]) ** 2).mean()

                    if self.args["shared"]:
                        actor_loss  = policy_loss - self.args["entropy_coeff"] * entropy.mean()
                        critic_loss = value_loss
                        loss        = actor_loss + self.args["value_coeff"] * critic_loss

                        episode_actor_losses.append(actor_loss.item())
                        episode_critic_losses.append(critic_loss.item())
                        episode_entropies.append(entropy.mean().item())
                        episode_total_losses.append(loss.item())

                        self.policy_optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.args["max_grad_norm"])
                        self.policy_optimizer.step()

                    else:
                        actor_loss  = policy_loss - self.args["entropy_coeff"] * entropy.mean()
                        critic_loss = value_loss

                        episode_actor_losses.append(actor_loss.item())
                        episode_critic_losses.append(critic_loss.item())
                        episode_entropies.append(entropy.mean().item())
                        episode_total_losses.append(0)

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args["max_grad_norm"])
                        self.actor_optimizer.step()

                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args["max_grad_norm"])
                        self.critic_optimizer.step()

            # Snapshot this update's mean actor/critic loss; episodes that
            # finish in the NEXT rollout record these values (per-episode
            # aligned loss log — see self.episode_actor_loss / _critic_loss).
            if len(episode_actor_losses) > _loss_start:
                self._last_actor_loss  = float(np.mean(episode_actor_losses[_loss_start:]))
                self._last_critic_loss = float(np.mean(episode_critic_losses[_loss_start:]))
                self._last_entropy     = float(np.mean(episode_entropies[_loss_start:]))

        return self.training_log()

    # ──────────────────────────────────────────────────────────────────────────
    # Inference / evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_episode(self, log_wandb=False, episode_idx=None):
        """
        Run one full episode with the deterministic (mean) policy.

        Returns (memory_deque, episode_df).

        episode_df columns:
            timestep, reward, value, distance,
            <nutrient_name> × N,
            <nutrient_name>_error × N,
            amount_slot_0 … amount_slot_{K-1},
            total_consumption,
            is_awake,
            cycle_number,
            time_in_cycle
        """
        env = self.env
        obs, _ = env.reset()
        done   = False

        memory  = deque(maxlen=env.max_steps)
        rewards, values, distances = [], [], []
        phy_states  = []    # nutrient dims only (no cycle dims)
        all_amounts = []    # (num_foods,) per step
        is_awake_log     = []
        cycle_number_log = []
        time_in_cycle_log = []

        while not done:
            amounts_t, log_prob, value, mu, log_std, x_raw = self.act(obs, deterministic=False)

            if self.is_continuous:
                amounts_np = amounts_t.squeeze(0).detach().cpu().numpy()
                env_action = amounts_np
            else:
                # amounts_t is a scalar action index in {0, ..., num_foods}.
                # Expand to the same (num_foods,) representation continuous
                # mode uses, so downstream logging/plotting code is unchanged.
                action_idx = int(amounts_t.squeeze(0).detach().cpu().item())
                amounts_np = np.zeros(env.num_foods, dtype=np.float32)
                if action_idx > 0:
                    amounts_np[action_idx - 1] = DISCRETE_EAT_AMOUNT
                env_action = action_idx

            next_obs, reward, terminated, done, info = env.step(env_action)

            # Extract nutrient dims only (drop the last 2 cycle dims)
            phy_full    = obs["physiological_state"]
            phy_nutri   = phy_full[:env.num_nutrients].copy()

            rewards.append(reward)
            values.append(value.item())
            phy_states.append(phy_nutri)
            all_amounts.append(amounts_np.copy())
            distances.append(float(np.linalg.norm(phy_nutri - env._norm_targets, ord=1)))
            is_awake_log.append(info["is_awake"])
            cycle_number_log.append(info["cycle_number"])
            time_in_cycle_log.append(info["time_in_cycle"])

            memory.append((
                obs["physiological_state"],   # full state incl cycle dims
                obs["food_embeddings"],
                amounts_np,
                reward,
                next_obs["physiological_state"],
                next_obs["food_embeddings"],
                mu,
                log_std,
                value,
                done,
            ))

            obs = next_obs

        # ── Build episode dataframe ────────────────────────────────────────
        phy_arr    = np.array(phy_states)    # (T, num_nutrients)
        target_arr = env._norm_targets
        names      = env.nutrient_names
        T          = len(rewards)

        col_dict = {
            "timestep":      np.arange(T),
            "reward":        np.array(rewards),
            "value":         np.array(values),
            "distance":      np.array(distances),
            "is_awake":      np.array(is_awake_log, dtype=bool),
            "cycle_number":  np.array(cycle_number_log, dtype=int),
            "time_in_cycle": np.array(time_in_cycle_log, dtype=np.float32),
        }

        for i, name in enumerate(names):
            col_dict[name]            = phy_arr[:, i]
            col_dict[f"{name}_error"] = np.abs(phy_arr[:, i] - target_arr[i])

        amounts_arr = np.array(all_amounts)  # (T, num_foods)
        for slot_i in range(env.num_foods):
            col_dict[f"amount_slot_{slot_i}"] = amounts_arr[:, slot_i]
        col_dict["total_consumption"] = amounts_arr.sum(axis=1)

        # ── Inference-only calorie accounting ──────────────────────────────
        # Amount-scaled calories eaten per step + running total, computed from
        # the env's per-food calorie totals (never part of obs/reward/training).
        cal_step = np.zeros(T, dtype=np.float64)
        for e in env._step_consumption:
            ts = int(e["timestep"])
            if 0 <= ts < T:
                cal_step[ts] += e["amount"] * env._calorie_totals.get(e["food_name"], 0.0)
        col_dict["calories_consumed"]   = cal_step
        col_dict["cumulative_calories"] = np.cumsum(cal_step)

        episode_df = pd.DataFrame(col_dict)

        if log_wandb:
            if not WANDB_AVAILABLE:
                warnings.warn("wandb not available – skipping episode logging.", RuntimeWarning)
            else:
                self._log_episode(episode_idx, rewards, values, phy_arr, distances, amounts_arr, episode_df, env)
                self._log_evaluation_summary(
                    episode_idx=episode_idx,
                    episode_df=episode_df,
                    returns=self.episode_returns,
                    env=env,
                )

        return memory, episode_df

    def infer_episode(self, memory):
        """
        Summarise last nutrient state vs target and total consumption per slot.
        Returns (nutrient_df, slot_df).
        """
        env    = self.env
        names  = env.nutrient_names
        target = env._norm_targets

        # Nutrient dims are [:num_nutrients] of the stored state
        last_phy = memory[-1][0][:env.num_nutrients]

        nutrient_df = pd.DataFrame({
            "Nutrient":        names,
            "Target (normed)": target,
            "Actual (normed)": last_phy,
        })

        total_consumed = np.zeros(env.num_foods, dtype=np.float32)
        for entry in memory:
            total_consumed += entry[2]  # amounts at each step

        slot_df = pd.DataFrame({
            "Slot":           [f"slot_{i}" for i in range(env.num_foods)],
            "Total Consumed": total_consumed,
        })

        return nutrient_df, slot_df

    # ──────────────────────────────────────────────────────────────────────────
    # WandB logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log_episode(self, episode_idx, rewards, values, phy_arr, distances, amounts_arr, episode_df, env):
        steps = np.arange(len(rewards))
        names = env.nutrient_names

        log_dict = {
            f"inference_{episode_idx}/reward": wandb.plot.line_series(
                xs=steps, ys=[np.array(rewards)],
                keys=["Reward"], title="Reward", xname="Timestep",
            ),
            f"inference_{episode_idx}/value": wandb.plot.line_series(
                xs=steps, ys=[np.array(values)],
                keys=["V(s)"], title="Critic value", xname="Timestep",
            ),
            f"inference_{episode_idx}/distance_to_target": wandb.plot.line_series(
                xs=steps, ys=[np.array(distances)],
                keys=["L1 distance"], title="Homeostatic error", xname="Timestep",
            ),
            f"inference_{episode_idx}/nutrient_states": wandb.plot.line_series(
                xs=steps,
                ys=[phy_arr[:, i] for i in range(env.num_nutrients)],
                keys=names,
                title="Physiological state (normalised)",
                xname="Timestep",
            ),
            f"inference_{episode_idx}/total_consumption": wandb.plot.line_series(
                xs=steps,
                ys=[amounts_arr.sum(axis=1)],
                keys=["Total consumption"],
                title="Total consumption per step",
                xname="Timestep",
            ),
            f"inference_{episode_idx}/per_slot_consumption": wandb.plot.line_series(
                xs=steps,
                ys=[amounts_arr[:, i] for i in range(env.num_foods)],
                keys=[f"slot_{i}" for i in range(env.num_foods)],
                title="Per-slot consumption amounts",
                xname="Timestep",
            ),
            # Log is_awake as a 0/1 signal so sleep phases are visible in wandb
            f"inference_{episode_idx}/is_awake": wandb.plot.line_series(
                xs=steps,
                ys=[episode_df["is_awake"].astype(float).values],
                keys=["is_awake"],
                title="Awake (1) / Sleep (0)",
                xname="Timestep",
            ),
        }

        wandb.log(log_dict)

        fig = env.plot_consumption(max_time=env.max_steps)
        wandb.log({f"inference_{episode_idx}/consumption_plot": wandb.Image(fig)})
        plt.close(fig)

    def _log_evaluation_summary(self, episode_idx, episode_df, returns, env):

        T = len(episode_df)

        # Nutrient state trajectory (T+1, N) — repeat last row as terminal placeholder
        states = np.vstack([
            episode_df[list(env.nutrient_names)].values,
            episode_df[list(env.nutrient_names)].iloc[[-1]].values,
        ])

        rewards      = episode_df["reward"].values
        distances    = episode_df["distance"].values
        is_awake_arr = episode_df["is_awake"].values  # (T,) bool

        amounts_arr = np.stack(
            [episode_df[f"amount_slot_{i}"].values for i in range(env.num_foods)],
            axis=1,
        )  # (T, num_foods)

        n           = env.num_nutrients
        names       = env.nutrient_names
        target_norm = env._norm_targets
        norm_low    = env._norm_target_low
        norm_high   = env._norm_target_high

        t_axis = np.arange(T + 1)
        t_step = np.arange(1, T + 1)

        # Unnormalise
        unnorm_states  = np.zeros_like(states)
        unnorm_targets = np.zeros(n, dtype=np.float32)
        unnorm_low     = np.zeros(n, dtype=np.float32)
        unnorm_high    = np.zeros(n, dtype=np.float32)
        for i, nm in enumerate(names):
            v_min = env._nutrient_mins[nm]
            v_max = env._nutrient_maxs[nm]
            rng   = v_max - v_min
            unnorm_states[:, i] = states[:, i] * rng + v_min
            unnorm_targets[i]   = target_norm[i] * rng + v_min
            unnorm_low[i]       = norm_low[i] * rng + v_min
            unnorm_high[i]      = norm_high[i] * rng + v_min

        # Sleep window boundaries from env
        sleep_windows = env.sleep_windows()

        SKIP_EPS = 15
        WINDOW   = 50
        n_rows   = n + 3

        fig = plt.figure(figsize=(14, 4.5 * n_rows))
        gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.5, wspace=0.35)

        # ── Training curve ────────────────────────────────────────────────
        ax_tr = fig.add_subplot(gs[0, :])
        plot_returns  = returns[SKIP_EPS:]
        plot_episodes = np.arange(SKIP_EPS, len(returns))
        rolling = np.convolve(plot_returns, np.ones(WINDOW) / WINDOW, mode="valid")
        roll_x  = np.arange(SKIP_EPS + WINDOW - 1, len(returns))
        ax_tr.plot(plot_episodes, plot_returns, alpha=0.3, color="steelblue", label="Episode return")
        ax_tr.plot(roll_x, rolling, color="steelblue", lw=2, label=f"Rolling avg ({WINDOW})")
        max_ret = float(np.max(returns))
        max_ep  = int(np.argmax(returns))
        ax_tr.scatter([max_ep], [max_ret], color="crimson", zorder=5, s=60)
        ax_tr.annotate(f"Max: {max_ret:.2f}", xy=(max_ep, max_ret),
                       xytext=(12, -18), textcoords="offset points",
                       color="crimson", fontsize=9,
                       arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
        ax_tr.set_xlabel("Episode")
        ax_tr.set_ylabel("Return")
        ax_tr.set_title(f"PPO Training Curve (first {SKIP_EPS} eps omitted)")
        ax_tr.legend(fontsize=9)
        ax_tr.grid(True, alpha=0.3)

        def _shade_sleep(ax, windows, t_max):
            """Helper: shade sleep windows on any axis."""
            for k, (ws, we) in enumerate(windows):
                ax.axvspan(ws, min(we, t_max), alpha=0.10, color="navy",
                           label="Sleep" if k == 0 else "")

        # ── Nutrient rows ─────────────────────────────────────────────────
        for i, nm in enumerate(names):
            row = i + 1

            ax_n = fig.add_subplot(gs[row, 0])
            ax_n.axhspan(norm_low[i], norm_high[i], alpha=0.15, color="crimson")
            ax_n.axhline(target_norm[i], ls="--", color="crimson", lw=1.0, alpha=0.5)
            ax_n.plot(t_axis, states[:, i], color="steelblue", lw=1.5, label="Agent state")
            ref_n = np.clip(states[:, i], norm_low[i], norm_high[i])
            ax_n.fill_between(t_axis, states[:, i], ref_n, alpha=0.12, color="steelblue")
            _shade_sleep(ax_n, sleep_windows, T)
            ax_n.set_title(f"{nm} — normalised")
            ax_n.grid(True, alpha=0.3)

            ax_u = fig.add_subplot(gs[row, 1])
            ax_u.axhspan(unnorm_low[i], unnorm_high[i], alpha=0.15, color="crimson")
            ax_u.axhline(unnorm_targets[i], ls="--", color="crimson", lw=1.0, alpha=0.5)
            ax_u.plot(t_axis, unnorm_states[:, i], color="darkorange", lw=1.5, label="Agent state")
            ref_u = np.clip(unnorm_states[:, i], unnorm_low[i], unnorm_high[i])
            ax_u.fill_between(t_axis, unnorm_states[:, i], ref_u, alpha=0.12, color="darkorange")
            _shade_sleep(ax_u, sleep_windows, T)
            ax_u.set_title(f"{nm} — raw units")
            ax_u.grid(True, alpha=0.3)

        # ── Reward / distance ─────────────────────────────────────────────
        ax_r = fig.add_subplot(gs[n + 1, 0])
        ax_r.plot(t_step, rewards, color="darkorange", lw=1.5)
        ax_r.axhline(0, color="grey", lw=0.8, ls="--")
        _shade_sleep(ax_r, sleep_windows, T)
        ax_r.set_title("Reward per step")
        ax_r.grid(True, alpha=0.3)

        ax_d = fig.add_subplot(gs[n + 1, 1])
        ax_d.plot(t_step, distances, color="teal", lw=1.5)
        _shade_sleep(ax_d, sleep_windows, T)
        ax_d.set_title("Distance to target")
        ax_d.grid(True, alpha=0.3)

        # ── Consumption heatmap / error heatmap ───────────────────────────
        ax_ch = fig.add_subplot(gs[n + 2, 0])
        im_c = ax_ch.imshow(
            amounts_arr.T,    # (num_foods, T)
            aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        ax_ch.set_yticks(range(env.num_foods))
        ax_ch.set_yticklabels([f"slot_{i}" for i in range(env.num_foods)])
        ax_ch.set_xlabel("Timestep")
        ax_ch.set_title("Consumption amounts (slot × time)")
        plt.colorbar(im_c, ax=ax_ch, label="Amount [0, 1]")

        below = np.minimum(0.0, states - norm_low[None, :])
        above = np.maximum(0.0, states - norm_high[None, :])
        error = below + above
        ax_h  = fig.add_subplot(gs[n + 2, 1])
        im    = ax_h.imshow(
            error.T, aspect="auto", cmap="RdBu_r",
            vmin=-np.abs(error).max(), vmax=np.abs(error).max(),
            interpolation="nearest",
        )
        ax_h.set_yticks(range(n))
        ax_h.set_yticklabels(names)
        ax_h.set_title("Signed nutrient error")
        plt.colorbar(im, ax=ax_h)

        fig.suptitle("PPO Agent — Evaluation Summary", fontsize=14, y=1.005)
        wandb.log({f"inference_{episode_idx}/evaluation_summary": wandb.Image(fig)})
        plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────────
    # Save / load
    # ──────────────────────────────────────────────────────────────────────────

    def _build_checkpoint_meta(self):
        """
        Assemble a self-contained checkpoint_meta dict: everything needed to
        rebuild the env AND the agent network with IDENTICAL tensor shapes,
        then load the weights and run inference — nothing else required.
        Consumed by PPOAgent.from_checkpoint(). Two source-of-truth blocks:

          meta["env"]   : kwargs to reconstruct FoodEnv
          meta["agent"] : kwargs to reconstruct PPOAgent (incl. the full
                          nested `bottlenecks` config)

        The remaining top-level keys are provenance / convenience (bottleneck
        names, last-set amplitudes, best-return bookkeeping) and are NOT
        required for reconstruction.
        """
        return {
            "env": {
                "food_folder":           self.env.food_folder,
                "num_foods":             self.env.num_foods,
                "max_steps":             self.env.max_steps,
                "awake_steps_per_cycle": self.env.awake_steps,
                "sleep_steps_per_cycle": self.env.sleep_steps,
                "is_continuous":         bool(self.env.is_continuous),
                "one_hot_embedding":     bool(self.env.one_hot_embedding),
                "embed_size":            self.env.args.get("embed_size"),
                "consumption_threshold": float(self.env.consumption_threshold),
                "seed":                  self.args["seed"],
            },
            "agent": {
                "shared":      self.args["shared"],
                "hidden":      self.args["hidden"],
                "bottlenecks": self.bottlenecks_config,
                "seed":        self.args["seed"],
            },
            # ── provenance / convenience (not needed to reconstruct) ──────
            "is_continuous":    bool(self.env.is_continuous),
            "seed":             self.args["seed"],
            # Flat convenience keys mirrored from env/agent for quick reference
            # and for analysis code that keys off them. `separate_branches` is
            # the representative (actor/policy) trunk's mode — the network
            # latent_analysis reads by default.
            "shared":           self.args["shared"],
            "hidden":           self.args["hidden"],
            "separate_branches": bool(getattr(
                self.policy if self.args["shared"] else self.actor,
                "separate_branches", False)),
            "awake_steps":      self.env.awake_steps,
            "sleep_steps":      self.env.sleep_steps,
            "num_foods":        self.env.num_foods,
            "max_steps":        self.env.max_steps,
            "bottleneck_names": list(self.bottlenecks.keys()),
            # Last amplitude set per bottleneck (informational only — runtime
            # gains are NOT restored on load; re-call set_amplitude after
            # from_checkpoint to reinstate a probe).
            "amplitudes":       dict(self._amplitude_record),
            "biases":           dict(self._bias_record),
            "best_return":      (None if self._best_episode is None
                                 else self._best_return),
            "best_episode":     self._best_episode,
        }

    def _seed_dir(self, checkpoint_dir):
        """Namespace all checkpoints for this run in a per-seed folder named
        '{checkpoint_dir}_{seed}' (e.g. checkpoint_dir='checkpoints', seed=42
        -> 'checkpoints_42'), so runs with different seeds never overwrite
        each other."""
        seed = self.args["seed"]
        tag = str(seed) if seed is not None else "none"
        return f"{checkpoint_dir.rstrip('/')}_{tag}"

    def _save_bundle(self, path, log_wandb=False):
        """Write the three-file bundle (weights + meta.json + log.npz) at the
        given path prefix. The *_log.npz holds the per-episode training log
        (reward, distance, consumption, actor_loss, critic_loss). Each bundle
        is independently loadable via from_checkpoint(path + '_meta.json')."""
        self.save_model(path, log_wandb=log_wandb)
        with open(path + "_meta.json", "w") as f:
            json.dump(self._build_checkpoint_meta(), f, indent=2)
        self.save_training_log(path + "_log.npz")
        return path

    def save_checkpoint(self, checkpoint_dir, episode_count, log_wandb=False):
        """
        Save a full, independently-loadable checkpoint bundle at the given
        episode count, in the per-seed folder '{checkpoint_dir}_{seed}':

            {checkpoint_dir}_{seed}/ppo_agent_ep{N}_shared.pt  (or _actor/_critic.pt)
            {checkpoint_dir}_{seed}/ppo_agent_ep{N}_meta.json
            {checkpoint_dir}_{seed}/ppo_agent_ep{N}_log.npz

        Reload any of them with PPOAgent.from_checkpoint(<prefix>_meta.json).
        """
        d = self._seed_dir(checkpoint_dir)
        os.makedirs(d, exist_ok=True)
        return self._save_bundle(
            os.path.join(d, f"ppo_agent_ep{episode_count}"), log_wandb=log_wandb)

    def save_best(self, checkpoint_dir, log_wandb=False):
        """
        Save/overwrite the dedicated BEST bundle (highest single-episode return
        seen so far) at {checkpoint_dir}_{seed}/ppo_agent_best_*. Called
        automatically during train() whenever a new best is hit and
        checkpointing is enabled; also callable manually.
        """
        d = self._seed_dir(checkpoint_dir)
        os.makedirs(d, exist_ok=True)
        return self._save_bundle(os.path.join(d, "ppo_agent_best"), log_wandb=log_wandb)

    def save_final(self, checkpoint_dir, log_wandb=False):
        """
        Save the end-of-training bundle at {checkpoint_dir}_{seed}/
        ppo_agent_last_*. This is the training→inference handoff artifact
        (replaces the notebook's hand-built checkpoint_meta cell).
        """
        d = self._seed_dir(checkpoint_dir)
        os.makedirs(d, exist_ok=True)
        return self._save_bundle(os.path.join(d, "ppo_agent_last"), log_wandb=log_wandb)

    @classmethod
    def from_checkpoint(cls, meta_path, device="cpu", weights_path=None,
                        **agent_overrides):
        """
        Rebuild env + agent from a checkpoint's meta.json and load its weights,
        in one call — ready for inference. `weights_path` (the save prefix,
        e.g. '.../ppo_agent_best') is inferred by stripping '_meta.json' from
        meta_path unless given explicitly. Extra **agent_overrides are passed
        to the PPOAgent constructor (e.g. PPO hyperparameters for further
        training); architecture/env come entirely from the meta.

        Returns the agent; the reconstructed env is available as agent.env.
        Amplitude probes are NOT restored (runtime-only) — the last-set values
        are recorded in meta['amplitudes']; re-call agent.set_amplitude(...) to
        reinstate them.
        """
        with open(meta_path) as f:
            meta = json.load(f)
        env_meta   = meta["env"]
        agent_meta = meta["agent"]

        env = FoodEnv(
            food_folder=env_meta["food_folder"],
            num_foods=env_meta["num_foods"],
            max_steps=env_meta["max_steps"],
            one_hot_embedding=env_meta.get("one_hot_embedding", True),
            embed_size=env_meta.get("embed_size"),
            seed=env_meta.get("seed"),
            consumption_threshold=env_meta.get("consumption_threshold", 0.0),
            awake_steps_per_cycle=env_meta["awake_steps_per_cycle"],
            sleep_steps_per_cycle=env_meta["sleep_steps_per_cycle"],
            is_continuous=env_meta["is_continuous"],
        )
        agent = cls(
            env, device=device,
            shared=agent_meta["shared"],
            hidden=agent_meta["hidden"],
            bottlenecks=agent_meta["bottlenecks"],
            seed=agent_meta.get("seed"),
            **agent_overrides,
        )

        if weights_path is None:
            if meta_path.endswith("_meta.json"):
                weights_path = meta_path[: -len("_meta.json")]
            else:
                raise ValueError(
                    "Could not infer weights_path from meta_path "
                    f"({meta_path!r} does not end in '_meta.json'); "
                    "pass weights_path=<save prefix> explicitly."
                )
        agent.load_model(weights_path)

        best = meta.get("best_return")
        agent._best_return  = best if best is not None else -float("inf")
        agent._best_episode = meta.get("best_episode")
        return agent

    def save_model(self, path, log_wandb=False):
        if self.args["shared"]:
            torch.save(self.policy.state_dict(), path + "_shared.pt")
        else:
            torch.save(self.actor.state_dict(),  path + "_actor.pt")
            torch.save(self.critic.state_dict(), path + "_critic.pt")

        if log_wandb:
            if not WANDB_AVAILABLE:
                warnings.warn("wandb not installed – skipping artifact save.", RuntimeWarning)
                return
            artifact = wandb.Artifact("PPOAgent", type="model")
            if self.args["shared"]:
                artifact.add_file(path + "_shared.pt")
            else:
                artifact.add_file(path + "_actor.pt")
                artifact.add_file(path + "_critic.pt")
            wandb.log_artifact(artifact)

    def load_model(self, path):
        if self.args["shared"]:
            self.policy.load_state_dict(torch.load(path + "_shared.pt"))
        else:
            self.actor.load_state_dict( torch.load(path + "_actor.pt"))
            self.critic.load_state_dict(torch.load(path + "_critic.pt"))