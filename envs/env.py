"""
bio_env.py  –  Biologically-grounded Food RL Environment
               with Continuous Actions and Sleep/Wake Cycles

─────────────────────────────────────────────────────────────────────────────
ACTION SPACE
    Controlled by the is_continuous constructor flag.

    is_continuous=True  (default, original behaviour)
        Box(num_foods,)  — each value in [0, 1]
        amounts[i] = 0.0  → do not eat food at slot i
        amounts[i] = 1.0  → eat full portion of food at slot i
        amounts[i] = 0.5  → eat half portion (absorption profile scaled by 0.5)

    is_continuous=False
        Discrete(num_foods + 1)  — a single categorical choice per step
        action == 0        → eat nothing this step
        action == k (k>=1) → eat from menu slot (k-1), at a FIXED fraction
                              (DISCRETE_EAT_AMOUNT, set in env_config.py).
        Only ONE slot can be eaten per step in discrete mode — there is no
        portion control and no multi-food meals within a single step.
        step() internally expands this into the same (num_foods,) amounts
        vector the continuous path uses, so all downstream absorption /
        decay / reward logic is identical regardless of mode.

    During SLEEP the agent still outputs an action, but the env IGNORES it
    entirely — no absorption occurs. The agent is penalised if it is
    nutritionally depleted while asleep (it can't eat to fix it).

─────────────────────────────────────────────────────────────────────────────
OBSERVATION SPACE
    {
        "physiological_state" : Box(num_nutrients + 2,)
            ├── nutrient levels [0 … num_nutrients-1]  (normalised)
            ├── is_awake   [-1]   1.0 = awake, 0.0 = asleep
            └── time_in_cycle [-1]  0.0 = phase just started → 1.0 = phase ending
        "food_embeddings"     : Box(num_foods, embed_size)
    }

    The two extra scalars let the agent anticipate phase transitions, e.g.
    "I'm awake and time_in_cycle=0.9 → sleep soon, I should eat now."

─────────────────────────────────────────────────────────────────────────────
SLEEP/WAKE CYCLE
    Each cycle = awake_steps_per_cycle + sleep_steps_per_cycle env steps.
    Default: 960 awake (32 hrs at 2 min/step) + 480 sleep (16 hrs) = 1440/cycle.

    A one-week episode = 14 awake phases + 14 sleep phases
    = 14 × (960 + 480) = 20,160 steps.  Use max_steps=20160 for weekly runs.

─────────────────────────────────────────────────────────────────────────────
NUTRIENT_CONFIG
    Each nutrient now has two extra keys:
        sleep_decay_rate     – decay applied during sleep (typically < decay_rate)
        sleep_penalty_weight – extra multiplier on the below-threshold penalty
                               during sleep (cannot eat to recover, so worse)

    Comment/uncomment nutrient blocks to activate/deactivate them.
    All other env logic adapts automatically.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import os
from collections import deque
from typing import Dict, List, Optional
from matplotlib import pyplot as plt

from envs.env_config import NUTRIENT_CONFIG, DISCRETE_EAT_AMOUNT, SHADOW_NUTRIENT_CONFIG


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Each env step covers this many real minutes.
MINUTES_PER_STEP: int = 5 #30 #2

# ── Inference-only calorie accounting ─────────────────────────────────────────
# Calories are loaded from this CSV purely to REPORT total calories consumed
# during inference episodes. They deliberately never enter NUTRIENT_CONFIG, the
# observation, or the reward — training is completely unaffected whether or not
# this file exists. calories_absorbed.csv is cumulative (monotonic per food), so
# the final row is that food's total absorbed calories for a full portion.
CALORIE_CSV: str = "calories_absorbed.csv"
CALORIE_COL_SUFFIX: str = "_calories absorbed"

# NUTRIENT_CONFIG and DISCRETE_EAT_AMOUNT now live in env_config.py — see that
# file to add/edit nutrients or change the fixed discrete eating amount.


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_normalise(filepath, col_suffix, is_cumulative):
    """Load a nutrient CSV and normalise values to [0, 1]."""
    ext = os.path.splitext(filepath)[1].lower()
    df  = pd.read_excel(filepath) if ext in (".xlsx", ".xls") else pd.read_csv(filepath)

    feature_df         = df.drop(columns=["time"]).fillna(0)
    food_names         = [c.replace(col_suffix, "").strip() for c in feature_df.columns]
    feature_df.columns = food_names

    if is_cumulative:
        C_i   = feature_df.iloc[-1].astype(float)
        v_min = float(C_i.min())
        v_max = float(C_i.max())
        denom = (v_max - v_min) if (v_max - v_min) > 0.0 else 1.0
        norm_data = (C_i - v_min) / denom
    else:
        v_min = float(feature_df.values.min())
        v_max = float(feature_df.values.max())
        denom = (v_max - v_min) if (v_max - v_min) > 0.0 else 1.0
        norm_data = (feature_df - v_min) / denom

    return food_names, norm_data, v_min, v_max


def _build_delta_profile(col, minutes_per_step):
    """Convert a normalised cumulative column into signed per-env-step deltas."""
    col        = col.astype(np.float32)
    indices    = np.arange(0, len(col), minutes_per_step)
    subsampled = col[indices]
    deltas     = np.diff(subsampled, prepend=np.float32(0.0))
    return deltas.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FoodEnv
# ══════════════════════════════════════════════════════════════════════════════

class FoodEnv(gym.Env):
    """
    Biologically-grounded food RL environment with continuous actions
    and a fixed sleep/wake cycle.

    Parameters
    ──────────
    food_folder            : str   – directory containing nutrient CSVs
    num_foods              : int   – menu size per step (K)
    max_steps              : int   – total episode length in steps
                                     (use 20160 for a full 7-day week)
    one_hot_embedding      : bool  – one-hot food embeddings
    embed_size             : int   – learned embedding dim (if not one-hot)
    seed                   : int   – RNG seed
    consumption_threshold  : float – amounts below this are treated as zero
    awake_steps_per_cycle  : int   – env steps the agent is awake each cycle
                                     default 960 = 32 hrs at 2 min/step
    sleep_steps_per_cycle  : int   – env steps the agent sleeps each cycle
                                     default 480 = 16 hrs at 2 min/step
    is_continuous          : bool  – True (default) → Box(num_foods,) action
                                     space, fractional consumption per slot.
                                     False → Discrete(num_foods + 1) action
                                     space; action 0 = eat nothing, action k
                                     (k>=1) = eat slot (k-1) at the fixed
                                     amount DISCRETE_EAT_AMOUNT (env_config.py).
    """

    metadata = {"render_modes": []}

    def __init__(self, food_folder: str, **args):
        super().__init__()

        defaults = dict(
            num_foods=5,
            max_steps=5040,          # 7 × (480 + 240) = one full week
            one_hot_embedding=True,
            embed_size=None,
            seed=0,
            consumption_threshold=0.1,
            awake_steps_per_cycle=480,   # 480 × 2 min = 960 min ≈ 16 hrs awake
            sleep_steps_per_cycle=240,   # 240 × 2 min =  480 min ≈ 8 hrs asleep
            is_continuous=True,
        )
        self.args = {**defaults, **args}
        unknown   = set(args) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown args: {unknown}")

        if self.args["embed_size"] is not None and self.args["one_hot_embedding"]:
            raise ValueError("Cannot use both one_hot_embedding and embed_size.")

        self.food_folder      = food_folder
        self.max_steps        = self.args["max_steps"]
        self.num_foods        = self.args["num_foods"]
        self.nutrient_names   = list(NUTRIENT_CONFIG.keys())
        self.num_nutrients    = len(self.nutrient_names)
        self.minutes_per_step = MINUTES_PER_STEP
        self.is_continuous    = bool(self.args["is_continuous"])

        if self.num_nutrients == 0:
            raise ValueError("NUTRIENT_CONFIG is empty — activate at least one nutrient.")

        # ── Sleep / wake cycle parameters ─────────────────────────────────────
        # A single cycle = awake phase + sleep phase.
        # The agent's position within the cycle is computed every step from
        # self.timepoint and exposed in the observation as (is_awake, time_in_cycle).
        self.awake_steps  = int(self.args["awake_steps_per_cycle"])
        self.sleep_steps  = int(self.args["sleep_steps_per_cycle"])
        self.cycle_length = self.awake_steps + self.sleep_steps

        # ── Consumption threshold ──────────────────────────────────────────────
        # Amounts below this value are zeroed — prevents micro-doses like 0.01
        # from registering as meaningful eating events.
        self.consumption_threshold = float(self.args["consumption_threshold"])

        self._nutrient_mins: Dict[str, float] = {}
        self._nutrient_maxs: Dict[str, float] = {}
        self._load_food_library()

        if self.num_foods > self.num_items:
            raise ValueError(
                f"num_foods={self.num_foods} exceeds available food items ({self.num_items})."
            )

        self.one_hot_embedding = self.args["one_hot_embedding"]
        self.embed_size = (
            self.num_items if self.one_hot_embedding else self.args["embed_size"]
        )
        self._build_food_embeddings()

        if not self.is_continuous and DISCRETE_EAT_AMOUNT < self.consumption_threshold:
            raise ValueError(
                f"DISCRETE_EAT_AMOUNT ({DISCRETE_EAT_AMOUNT}) is below "
                f"consumption_threshold ({self.consumption_threshold}); every "
                "discrete 'eat' action would be silently zeroed out. Lower "
                "consumption_threshold or raise DISCRETE_EAT_AMOUNT in "
                "env_config.py."
            )

        # ── Normalised targets and range boundaries ────────────────────────────
        def _norm(val, n):
            # No clipping to [0, 1]. The agent's physiological state is itself
            # unclipped (the leaky integrator can exceed 1.0), so targets and
            # band edges are kept on the SAME raw normalised scale. A target
            # above the data max therefore yields a value > 1.0 (the band sits
            # above the reachable range and stays visible/measurable) instead of
            # collapsing onto 1.0; likewise a target below the data min can go
            # negative.
            rng = max(self._nutrient_maxs[n] - self._nutrient_mins[n], 1e-8)
            return float((val - self._nutrient_mins[n]) / rng)

        self._norm_targets = np.array(
            [_norm(NUTRIENT_CONFIG[n]["target"], n) for n in self.nutrient_names],
            dtype=np.float32,
        )
        self._norm_target_low = np.array(
            [_norm(NUTRIENT_CONFIG[n]["target"] - NUTRIENT_CONFIG[n].get("tolerance", 0.0), n)
             for n in self.nutrient_names],
            dtype=np.float32,
        )
        self._norm_target_high = np.array(
            [_norm(NUTRIENT_CONFIG[n]["target"] + NUTRIENT_CONFIG[n].get("tolerance", 0.0), n)
             for n in self.nutrient_names],
            dtype=np.float32,
        )

        # ── Initial physiological levels ───────────────────────────────────────
        # Optional per-nutrient `initial_value` (in the SAME raw data units as
        # `target`) lets an episode start at a chosen level instead of an empty
        # integrator.
        #   key ABSENT  → 0.0  (empty integrator, phys state starts at 0.0) —
        #                 the original behaviour, so existing configs are
        #                 unaffected.
        #   key PRESENT → normalised to the phys-state scale exactly like the
        #                 target. Setting initial_value slightly below target
        #                 therefore starts the agent just under its band.
        # Only time-series nutrients are seedable; cumulative nutrients (e.g.
        # calories) are running totals and always start at 0.0.
        def _init_phys(n):
            iv = NUTRIENT_CONFIG[n].get("initial_value", None)
            return 0.0 if iv is None else _norm(iv, n)

        self._init_phys_ts: Dict[str, float] = {
            n: float(_init_phys(n)) for n in self._ts_nutrient_names
        }

        # ── Observation space ──────────────────────────────────────────────────
        # state_dim = num_nutrients + 2 (is_awake, time_in_cycle)
        # The +2 lets the network learn anticipatory behaviour:
        #   is_awake=1, time_in_cycle→1  →  sleep is approaching, eat now
        #   is_awake=0, time_in_cycle→1  →  waking soon, prepare for next day
        self.state_dim = self.num_nutrients + 2

        self.observation_space = spaces.Dict({
            "physiological_state": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim,), dtype=np.float32,
            ),
            "food_embeddings": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_foods, self.embed_size), dtype=np.float32,
            ),
        })

        # Action space depends on is_continuous:
        #   continuous → Box(num_foods,), fractional consumption per slot.
        #   discrete   → Discrete(num_foods + 1); 0 = eat nothing, k>=1 = eat
        #                slot (k-1) at the fixed DISCRETE_EAT_AMOUNT.
        if self.is_continuous:
            self.action_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_foods,), dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(self.num_foods + 1)

        self._seed = None
        if self.args["seed"] is not None:
            self._set_seed(self.args["seed"])

        self.reset(seed=self.args["seed"])

    # ──────────────────────────────────────────────────────────────────────────
    # Cycle helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _cycle_state(self, timepoint: int):
        """
        Compute sleep/wake context for a given absolute timepoint.

        Returns
        ───────
        is_awake       : bool   – True if within the awake phase of the cycle
        time_in_cycle  : float  – progress through the CURRENT phase, 0.0→1.0
                                  0.0 = phase just started
                                  1.0 = phase about to end
        cycle_number   : int    – which full cycle we are in (0-indexed day)
        """
        step_in_cycle = timepoint % self.cycle_length
        cycle_number  = timepoint // self.cycle_length
        is_awake      = step_in_cycle < self.awake_steps

        if is_awake:
            # Progress through the awake phase
            phase_len     = self.awake_steps
            phase_step    = step_in_cycle
        else:
            # Progress through the sleep phase
            phase_len     = self.sleep_steps
            phase_step    = step_in_cycle - self.awake_steps

        # Normalise to [0, 1]; avoid division by zero for single-step phases
        time_in_cycle = phase_step / max(phase_len - 1, 1)

        return is_awake, float(time_in_cycle), int(cycle_number)

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_food_library(self):
        per_nutrient: List[Dict] = []

        for n, cfg in NUTRIENT_CONFIG.items():
            filepath = os.path.join(self.food_folder, cfg["csv"])
            print(f"[FoodEnv] Loading  '{n}'  from  '{filepath}'")

            food_names, norm_data, v_min, v_max = _load_and_normalise(
                filepath, cfg["col_suffix"], cfg["is_cumulative"],
            )
            self._nutrient_mins[n] = v_min
            self._nutrient_maxs[n] = v_max

            per_nutrient.append({
                "name":          n,
                "food_names":    food_names,
                "data":          norm_data,
                "is_cumulative": cfg["is_cumulative"],
            })
            print(
                f"           min={v_min:.4f}  max={v_max:.4f}   foods={len(food_names)}"
                + (f"   time_points={norm_data.shape[0]}" if not cfg["is_cumulative"] else "")
            )

        common = set(per_nutrient[0]["food_names"])
        for entry in per_nutrient[1:]:
            common &= set(entry["food_names"])
        common = sorted(common)

        if not common:
            raise RuntimeError("No food items common across all active nutrient CSVs.")

        dropped = set(per_nutrient[0]["food_names"]) - set(common)
        if dropped:
            print(f"[FoodEnv] WARNING – dropping foods absent from some CSVs: {dropped}")

        self.item_list  = common
        self.num_items  = len(common)

        self._ts_nutrient_names    = [e["name"] for e in per_nutrient if not e["is_cumulative"]]
        self._cumul_nutrient_names = [e["name"] for e in per_nutrient if e["is_cumulative"]]

        ts_entries    = [e for e in per_nutrient if not e["is_cumulative"]]
        cumul_entries = [e for e in per_nutrient if e["is_cumulative"]]

        self._profiles:        Dict[str, np.ndarray] = {}
        self._calorie_scalars: Dict[str, Dict[str, float]] = {}

        for food in common:
            if ts_entries:
                cols = []
                for entry in ts_entries:
                    raw_col  = entry["data"][food].values.astype(np.float32)
                    step_col = _build_delta_profile(raw_col, self.minutes_per_step)
                    cols.append(step_col)

                T_max  = max(len(c) for c in cols)
                padded = np.stack(
                    [np.concatenate([c, np.zeros(T_max - len(c), dtype=np.float32)])
                     for c in cols],
                    axis=1,
                )
                self._profiles[food] = padded

            scalars = {}
            for entry in cumul_entries:
                scalars[entry["name"]] = float(entry["data"][food])
            self._calorie_scalars[food] = scalars

        print(
            f"[FoodEnv] Ready — {self.num_items} foods | "
            f"{len(self._ts_nutrient_names)} time-series nutrients | "
            f"{len(self._cumul_nutrient_names)} cumulative nutrients"
        )

        # ── Shadow nutrients (fullness / hunger, etc.) ─────────────────────────
        # Loaded and profiled with the exact same pipeline as the real
        # time-series nutrients above, but kept in entirely separate
        # attributes (_shadow_* prefix) so there is no path by which they
        # could leak into item_list, num_items, observation_space, or
        # reward. Restricted to `common` (the food set already fixed by the
        # real nutrients) — a shadow CSV is never allowed to expand or shrink
        # the action space; it can only attach (or fail to attach) a signal
        # to foods that already exist.
        self._shadow_mins: Dict[str, float] = {}
        self._shadow_maxs: Dict[str, float] = {}
        self.shadow_nutrient_names = list(SHADOW_NUTRIENT_CONFIG.keys())
        self._shadow_profiles: Dict[str, np.ndarray] = {}

        if self.shadow_nutrient_names:
            shadow_entries = []
            for n, cfg in SHADOW_NUTRIENT_CONFIG.items():
                filepath = os.path.join(self.food_folder, cfg["csv"])
                print(f"[FoodEnv] Loading shadow nutrient '{n}'  from  '{filepath}'")

                food_names, norm_data, v_min, v_max = _load_and_normalise(
                    filepath, cfg["col_suffix"], cfg["is_cumulative"],
                )
                self._shadow_mins[n] = v_min
                self._shadow_maxs[n] = v_max

                missing = set(common) - set(food_names)
                if missing:
                    print(
                        f"[FoodEnv] WARNING – shadow nutrient '{n}' has no data for "
                        f"{len(missing)} food(s) already in item_list: {sorted(missing)}. "
                        "These foods will contribute a zero profile for this shadow "
                        "nutrient (real nutrients / training are unaffected either way)."
                    )

                shadow_entries.append({
                    "name": n, "food_names": food_names, "data": norm_data,
                })

            for food in common:
                cols = []
                for entry in shadow_entries:
                    if food in entry["food_names"]:
                        raw_col = entry["data"][food].values.astype(np.float32)
                    else:
                        raw_col = np.zeros(1, dtype=np.float32)
                    step_col = _build_delta_profile(raw_col, self.minutes_per_step)
                    cols.append(step_col)

                T_max = max(len(c) for c in cols)
                padded = np.stack(
                    [np.concatenate([c, np.zeros(T_max - len(c), dtype=np.float32)])
                     for c in cols],
                    axis=1,
                )
                self._shadow_profiles[food] = padded

            print(
                f"[FoodEnv] Shadow nutrients ready — "
                f"{len(self.shadow_nutrient_names)} signal(s): {self.shadow_nutrient_names} "
                "(observation/reward untouched)"
            )

        # ── Inference-only calorie totals (observation/reward untouched) ───────
        self._load_calorie_totals()

    def _load_calorie_totals(self):
        """
        Load per-food TOTAL calories (raw kcal for a full portion) from
        CALORIE_CSV, restricted to foods already in item_list. Cumulative CSV →
        the final row is the total absorbed. Stored in self._calorie_totals and
        used ONLY to report calories consumed during inference (see
        episode_calories_consumed / calorie_summary / plot_calories). Never
        enters item_list, observation, or reward — if the file is absent,
        calorie totals are all 0.0 and everything else works unchanged.
        """
        self._calorie_totals: Dict[str, float] = {food: 0.0 for food in self.item_list}
        self._has_calories = False

        filepath = os.path.join(self.food_folder, CALORIE_CSV)
        if not os.path.exists(filepath):
            print(f"[FoodEnv] No calorie file at '{filepath}' — "
                  "calorie reporting disabled (totals = 0).")
            return

        ext = os.path.splitext(filepath)[1].lower()
        df  = pd.read_excel(filepath) if ext in (".xlsx", ".xls") else pd.read_csv(filepath)
        feature_df         = df.drop(columns=["time"]).fillna(0)
        names              = [c.replace(CALORIE_COL_SUFFIX, "").strip() for c in feature_df.columns]
        feature_df.columns = names

        # Per-food total = COLUMN MAX, not the final row.
        #
        # Most foods' curves are cumulative (monotonically rising), so max == final
        # row and the two agree. But a handful of fast-absorbing items (starch,
        # biscuit, glucose, ...) plateau and are then ZERO-FILLED for the rest of
        # the series once absorption completes — for those the final row is 0 even
        # though the food delivered hundreds of kcal. Taking the max recovers the
        # true total for those and is identical for every genuinely cumulative food.
        totals = feature_df.astype(float).max(axis=0)

        missing, truncated = [], []
        for food in self.item_list:
            if food in totals.index:
                self._calorie_totals[food] = float(totals[food])
                col = feature_df[food].astype(float)
                if float(col.iloc[-1]) < float(col.max()):
                    truncated.append(food)
            else:
                missing.append(food)
        if missing:
            print(f"[FoodEnv] WARNING – no calorie data for {len(missing)} food(s): "
                  f"{sorted(missing)} (counted as 0 kcal).")
        if truncated:
            print(f"[FoodEnv] NOTE – {len(truncated)} food(s) have non-cumulative "
                  f"(zero-filled after absorption) calorie curves: {sorted(truncated)}. "
                  "Using the curve MAX as their total.")

        self._has_calories = True
        n_have = sum(v > 0 for v in self._calorie_totals.values())
        n_tiny = sum(0 < v < 1.0 for v in self._calorie_totals.values())
        print(f"[FoodEnv] Calorie totals ready — {n_have}/{self.num_items} foods "
              "(inference-only; observation/reward untouched)")
        if n_tiny:
            print(f"[FoodEnv] NOTE – {n_tiny} food(s) have a total below 1 kcal, which "
                  "is likely missing/!=kcal source data rather than real energy content.")

    # ──────────────────────────────────────────────────────────────────────────
    # Seeding / embeddings / menu
    # ──────────────────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int):
        self._seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_food_embeddings(self):
        if self.one_hot_embedding:
            self._all_embeddings = np.eye(self.num_items, dtype=np.float32)
        else:
            emb = nn.Embedding(self.num_items, self.embed_size)
            emb.weight.requires_grad_(False)
            self._all_embeddings = emb.weight.detach().cpu().numpy().astype(np.float32)

    def _sample_menu(self):
        self._menu = self.np_random.choice(
            self.num_items, size=self.num_foods, replace=False
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, target_loc=None):
        super().reset(seed=seed)
        if seed is not None:
            self._set_seed(seed)

        # Active foods being digested: each entry = {food_name, eat_time, amount}
        self._active_foods: List[Dict] = []

        # Full episode consumption log (never pruned) — used by plot_consumption.
        # Each entry: {food_name, eat_time, amount, profile}
        self._consumption_log: List[Dict] = []

        # Cumulative totals for non-rolling nutrients (e.g. calories)
        self._cumul_values: Dict[str, float] = {
            n: 0.0 for n in self._cumul_nutrient_names
        }

        # Leaky-integrator internal state per time-series nutrient.
        # Seeded so the OBSERVED phys state starts at each nutrient's configured
        # initial_value (0.0 by default). Since phys_state = sum(rolling buffer
        # of length window_size), an initial phys level V requires each of the w
        # buffer slots — and the internal state — to hold V / w (window-correct;
        # holds for any window_size, not just 1).
        self._internal_state: Dict[str, float] = {}
        for n in self._ts_nutrient_names:
            w = NUTRIENT_CONFIG[n]["window_size"]
            self._internal_state[n] = self._init_phys_ts[n] / w

        # ── Shadow nutrient state (fullness / hunger) ──────────────────────
        # Mirrors _internal_state above exactly, but kept fully separate.
        # _shadow_nutrients is the per-step readout consumed only by
        # latent_analysis.py / shadow_summary() — never by _get_obs().
        #
        # Initial values: each shadow nutrient starts at its configured
        # `initial_value` (SHADOW_NUTRIENT_CONFIG), defaulting to 0.0. Unlike the
        # real nutrients, a shadow `initial_value` is on the integrator scale
        # directly — shadow nutrients have no target / raw-unit framing, so there
        # is nothing to normalise against. Shadow nutrients also have no rolling
        # window, so the value is used as-is.
        self._shadow_internal_state: Dict[str, float] = {
            n: float(SHADOW_NUTRIENT_CONFIG[n].get("initial_value", 0.0))
            for n in self.shadow_nutrient_names
        }
        self._shadow_nutrients: Dict[str, float] = dict(self._shadow_internal_state)
        self._shadow_log: List[Dict] = []

        # Rolling buffers (each stores IS values; phys_state = sum of buffer).
        # Pre-filled with V / w so the buffer already sums to the seeded initial
        # phys level V on the very first observation (see _internal_state above).
        self._rolling_buffers: Dict[str, deque] = {}
        for n in self._ts_nutrient_names:
            w   = NUTRIENT_CONFIG[n]["window_size"]
            is0 = self._init_phys_ts[n] / w
            self._rolling_buffers[n] = deque([is0] * w, maxlen=w)

        # Physiological state vector — nutrients only (no cycle scalars yet).
        # Seeded to each nutrient's initial phys level so the FIRST observation
        # is consistent with the seeded internal state / buffers above.
        # Cumulative nutrients are not in _init_phys_ts → default to 0.0.
        self._phys_nutrients = np.array(
            [self._init_phys_ts.get(n, 0.0) for n in self.nutrient_names],
            dtype=np.float32,
        )

        self.timepoint = 0

        # Per-step consumption log for infer_episode / summaries
        self._step_consumption: List[Dict] = []

        # Track per-step cycle state for plotting sleep windows later
        # Each entry: {timestep, is_awake, cycle_number}
        self._cycle_log: List[Dict] = []

        self._sample_menu()
        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Execute one environment step.

        Parameters
        ----------
        action : depends on is_continuous.
            continuous : (num_foods,) float array in [0, 1].
                         Values below consumption_threshold are zeroed.
            discrete   : int (or 0-d array / array of length 1) in
                         {0, ..., num_foods}.
                         0      → eat nothing this step.
                         k>=1   → eat from menu slot (k-1), at the fixed
                                  DISCRETE_EAT_AMOUNT (see env_config.py).
            In both modes, action is ignored entirely during sleep — no
            absorption occurs.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        if self.is_continuous:
            amounts = np.asarray(action, dtype=np.float32)
            assert amounts.shape == (self.num_foods,), (
                f"Expected amounts shape ({self.num_foods},), got {amounts.shape}"
            )
        else:
            # Accept a bare int, numpy scalar, or single-element array/tensor.
            discrete_action = int(np.asarray(action).reshape(-1)[0])
            assert 0 <= discrete_action <= self.num_foods, (
                f"Expected discrete action in [0, {self.num_foods}], "
                f"got {discrete_action}"
            )
            amounts = np.zeros(self.num_foods, dtype=np.float32)
            if discrete_action > 0:
                slot_i = discrete_action - 1
                amounts[slot_i] = DISCRETE_EAT_AMOUNT

        # ── Determine current cycle phase ─────────────────────────────────────
        # This is computed BEFORE incrementing timepoint so that the decay and
        # reward at this step correctly reflect the phase the agent is in now.
        is_awake, time_in_cycle, cycle_number = self._cycle_state(self.timepoint)

        if not is_awake:
            # sleeping
            amounts = np.zeros(self.num_foods, dtype=np.float32)

        # Log cycle state for post-episode analysis (e.g. shading sleep windows)
        self._cycle_log.append({
            "timestep":     self.timepoint,
            "is_awake":     is_awake,
            "cycle_number": cycle_number,
        })

        # ── 1. Food intake (only during awake phase) ──────────────────────────
        # During sleep the agent's outputs are discarded — the gut is effectively
        # closed.  Any food still being digested from before sleep continues to
        # release its absorption profile (active_foods is not cleared at sleep).
        if is_awake:
            for slot_i, amount in enumerate(amounts):
                # Not an eating event: a zero (or sub-threshold) amount means
                # the agent chose NOT to eat this slot, so it must not create an
                # absorption / consumption-log entry. The explicit <= 0 guard
                # matters when consumption_threshold == 0.0, where a bare
                # "eat nothing" action (amount 0.0) would otherwise slip through
                # `amount < threshold` and be miscounted as a food eaten.
                if float(amount) <= 0.0 or float(amount) < self.consumption_threshold:
                    continue   # not eaten → treat as zero

                food_idx  = int(self._menu[slot_i])
                food_name = self.item_list[food_idx]

                self._active_foods.append({
                    "food_idx":  food_idx,
                    "food_name": food_name,
                    "eat_time":  self.timepoint,
                    "amount":    float(amount),
                })

                # Full-episode log for plot_consumption
                self._consumption_log.append({
                    "food_name": food_name,
                    "eat_time":  self.timepoint,
                    "amount":    float(amount),
                    "profile":   self._profiles[food_name],
                })

                # Per-step log for consumption_summary / generate_episode
                self._step_consumption.append({
                    "timestep":  self.timepoint,
                    "food_name": food_name,
                    "amount":    float(amount),
                })

                # Cumulative nutrients (e.g. calories) scaled by amount
                for n in self._cumul_nutrient_names:
                    self._cumul_values[n] += (
                        self._calorie_scalars[food_name].get(n, 0.0) * float(amount)
                    )

        # ── 2. Sum amount-scaled active absorption profiles ───────────────────
        # Foods eaten BEFORE sleep continue digesting through the night.
        # Their profiles are still stepped forward — only NEW eating is blocked.
        x_t          = np.zeros(len(self._ts_nutrient_names), dtype=np.float32)
        still_active = []

        for food in self._active_foods:
            age     = self.timepoint - food["eat_time"]
            profile = self._profiles.get(food["food_name"])

            if profile is not None and age < profile.shape[0]:
                x_t += profile[age] * food["amount"]
                still_active.append(food)
            # Food whose profile has run its course is dropped silently

        self._active_foods = still_active

        # ── 3. Leaky integrator with phase-dependent decay ────────────────────
        # During awake:  use decay_rate       (normal metabolic clearance)
        # During sleep:  use sleep_decay_rate (slower overnight clearance)
        # This creates realistic overnight nutrient dynamics — levels fall
        # more slowly but still drop, so the agent must have eaten enough
        # before bed to last through the sleep phase.
        phys_state_ts: Dict[str, float] = {}
        for i, n in enumerate(self._ts_nutrient_names):
            cfg = NUTRIENT_CONFIG[n]
            dr  = cfg["decay_rate"] if is_awake else cfg["sleep_decay_rate"]

            is_n = self._internal_state[n] * (1.0 - dr) + float(x_t[i])
            self._internal_state[n] = is_n

            self._rolling_buffers[n].append(is_n)
            phys_state_ts[n] = float(sum(self._rolling_buffers[n]))

        # ── 3b. Shadow nutrients (fullness / hunger) — observation/reward untouched ─
        # Exactly the same mechanics as step 3 above (leaky integrator,
        # phase-dependent decay), reusing the SAME self._active_foods list
        # that was already pruned in step 2 — so fullness/hunger see the
        # identical set of in-flight meals, at the identical ages, as
        # glucose/peptides/fatty_acids do. The only thing that differs is
        # which profile dict and which decay rates are used.
        #
        # Result lives in self._shadow_nutrients / self._shadow_log only.
        # Nothing below this block reads from _get_obs(), observation_space,
        # or the reward calculation — this is intentionally a dead end for
        # the agent, and a data source for latent_analysis.py only.
        if self.shadow_nutrient_names:
            x_shadow = np.zeros(len(self.shadow_nutrient_names), dtype=np.float32)
            for food in self._active_foods:
                age     = self.timepoint - food["eat_time"]
                profile = self._shadow_profiles.get(food["food_name"])
                if profile is not None and age < profile.shape[0]:
                    x_shadow += profile[age] * food["amount"]
                # else: profile exhausted — silently contributes nothing,
                # same convention as step 2's real-nutrient handling.

            for i, n in enumerate(self.shadow_nutrient_names):
                cfg = SHADOW_NUTRIENT_CONFIG[n]
                dr  = cfg["decay_rate"] if is_awake else cfg["sleep_decay_rate"]

                is_n = self._shadow_internal_state[n] * (1.0 - dr) + float(x_shadow[i])
                self._shadow_internal_state[n] = is_n
                self._shadow_nutrients[n] = is_n

            self._shadow_log.append({
                "timestep": self.timepoint,
                **{n: self._shadow_nutrients[n] for n in self.shadow_nutrient_names},
            })

        # ── 4. Assemble nutrient state vector (nutrients only, no cycle dims) ─
        nutrient_values = []
        for n in self.nutrient_names:
            if n in phys_state_ts:
                nutrient_values.append(phys_state_ts[n])
            else:
                nutrient_values.append(self._cumul_values.get(n, 0.0))

        self._phys_nutrients = np.array(nutrient_values, dtype=np.float32)

        # ── 5. Reward ─────────────────────────────────────────────────────────
        #
        # Awake reward (same as before):
        #   inside [low, high]  →  +weight × in_range_bonus
        #   below low           →  -10 × weight × (low - state)²
        #   above high          →  -10 × weight × (state - high)²
        #
        # Sleep reward (same structure, but):
        #   below low  →  -10 × weight × sleep_penalty_weight × (low - state)²
        #     The extra sleep_penalty_weight makes depletion during sleep more
        #     costly — the agent cannot eat to recover, so it should have
        #     planned ahead before going to sleep.
        #   above high and in-range bonuses are unchanged during sleep.
        reward = 0.0
        for i, n in enumerate(self.nutrient_names):
            cfg       = NUTRIENT_CONFIG[n]
            state_val = float(self._phys_nutrients[i])
            low       = float(self._norm_target_low[i])
            high      = float(self._norm_target_high[i])
            weight    = cfg["reward_weight"]
            bonus     = cfg.get("in_range_bonus", 0.0)

            if low <= state_val <= high:
                # In range — positive reward regardless of sleep/wake
                reward += weight * bonus

            elif state_val > high:
                # Above acceptable range — same penalty awake or asleep
                dist    = state_val - high
                reward += 2 * weight * (-(dist ** 2)) #10

            else:
                # Below acceptable range
                dist = low - state_val
                if is_awake:
                    # Normal penalty — agent can eat to fix this
                    reward += 2 * weight * (-(dist ** 2))
                else:
                    # Amplified sleep penalty — agent cannot eat to recover
                    sleep_pw = cfg.get("sleep_penalty_weight", 1.0)
                    reward   += 2 * weight * sleep_pw * (-(dist ** 2))

        # ── 6. Bookkeeping ────────────────────────────────────────────────────
        self.timepoint += 1
        terminated      = False
        truncated       = self.timepoint >= self.max_steps
        self._sample_menu()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────────────────────────────────
    # Observation / info
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        """
        Build the observation dict.

        physiological_state = [nutrient_0, …, nutrient_{N-1}, is_awake, time_in_cycle]
            is_awake      : 1.0 = currently awake, 0.0 = currently asleep
            time_in_cycle : 0.0 = phase just started, 1.0 = phase about to end

        The cycle signals are appended AFTER the nutrient dims so that the
        nutrient slice [:num_nutrients] remains stable for indexing.
        """
        is_awake, time_in_cycle, _ = self._cycle_state(self.timepoint)

        # Concatenate nutrients with cycle context
        full_state = np.concatenate([
            self._phys_nutrients,
            np.array([float(is_awake), time_in_cycle], dtype=np.float32),
        ])

        return {
            "physiological_state": full_state,
            "food_embeddings":     self._all_embeddings[self._menu].copy(),
        }

    def _get_info(self):
        # Distance = range-clipped L1 error (0 inside the zone, >0 outside)
        below = np.maximum(0.0, self._norm_target_low  - self._phys_nutrients)
        above = np.maximum(0.0, self._phys_nutrients - self._norm_target_high)
        range_distance = float(np.sum(below + above))

        is_awake, time_in_cycle, cycle_number = self._cycle_state(self.timepoint)

        return {
            "distance":       range_distance,
            "menu":           self._menu.copy(),
            "menu_names":     [self.item_list[i] for i in self._menu],
            "timepoint":      self.timepoint,
            "real_minutes":   self.timepoint * self.minutes_per_step,
            "is_awake":       is_awake,
            "cycle_number":   cycle_number,
            "time_in_cycle":  time_in_cycle,
            # Per-step consumption detail (empty during sleep)
            "step_consumed":  [
                e for e in self._step_consumption
                if e["timestep"] == self.timepoint - 1
            ],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Summary helpers
    # ──────────────────────────────────────────────────────────────────────────

    def nutrient_norm_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "nutrient":            self.nutrient_names,
            "data_min":            [self._nutrient_mins[n] for n in self.nutrient_names],
            "data_max":            [self._nutrient_maxs[n] for n in self.nutrient_names],
            "raw_target":          [NUTRIENT_CONFIG[n]["target"] for n in self.nutrient_names],
            "normalised_target":   list(self._norm_targets),
            "window_size":         [NUTRIENT_CONFIG[n]["window_size"] for n in self.nutrient_names],
            "decay_rate":          [NUTRIENT_CONFIG[n]["decay_rate"] for n in self.nutrient_names],
            "sleep_decay_rate":    [NUTRIENT_CONFIG[n]["sleep_decay_rate"] for n in self.nutrient_names],
            "sleep_penalty_weight":[NUTRIENT_CONFIG[n]["sleep_penalty_weight"] for n in self.nutrient_names],
            "reward_weight":       [NUTRIENT_CONFIG[n]["reward_weight"] for n in self.nutrient_names],
        })

    def food_profile_summary(self) -> pd.DataFrame:
        rows = []
        for food in self.item_list:
            row = {"food": food}
            if food in self._profiles:
                profile = self._profiles[food]
                row["profile_steps"] = profile.shape[0]
                for j, n in enumerate(self._ts_nutrient_names):
                    row[f"total_{n}"] = float(profile[:, j].sum())
            for n in self._cumul_nutrient_names:
                row[f"total_{n}"] = self._calorie_scalars[food].get(n, 0.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def shadow_summary(self) -> pd.DataFrame:
        """
        Per-step fullness/hunger (or whatever else is in SHADOW_NUTRIENT_CONFIG),
        one row per env step taken so far this episode. Observation/reward
        never touch this — it's purely for post-hoc analysis
        (see latent_analysis.py's capture_trunk_episode).
        """
        if not self.shadow_nutrient_names:
            return pd.DataFrame(columns=["timestep"])
        if not self._shadow_log:
            return pd.DataFrame(columns=["timestep", *self.shadow_nutrient_names])
        return pd.DataFrame(self._shadow_log)

    def consumption_summary(self) -> pd.DataFrame:
        """Return a DataFrame of every eating event: [timestep, food_name, amount]."""
        if not self._step_consumption:
            return pd.DataFrame(columns=["timestep", "food_name", "amount"])
        return pd.DataFrame(self._step_consumption)

    # ──────────────────────────────────────────────────────────────────────────
    # Inference-only calorie reporting (never touches observation / reward)
    # ──────────────────────────────────────────────────────────────────────────

    def episode_calories_consumed(self) -> float:
        """
        Total calories consumed so far this episode = Σ over every awake eating
        event of (portion amount × that food's total absorbed calories). Purely
        an inference readout, computed from the consumption log — it has no
        effect on training, observation, or reward.
        """
        return float(sum(
            e["amount"] * self._calorie_totals.get(e["food_name"], 0.0)
            for e in self._step_consumption
        ))

    def calorie_summary(self) -> pd.DataFrame:
        """
        Per-step calorie DataFrame over the episode taken so far, with columns:
            timestep, calories_consumed, cumulative_calories
        `calories_consumed` is the amount-scaled calories eaten AT that step;
        `cumulative_calories` is the running total (what you have "by end of
        day"). One row per env step, including steps where nothing was eaten.
        """
        T = int(self.timepoint)
        per_step = np.zeros(max(T, 0), dtype=np.float64)
        for e in self._step_consumption:
            ts = int(e["timestep"])
            if 0 <= ts < T:
                per_step[ts] += e["amount"] * self._calorie_totals.get(e["food_name"], 0.0)
        return pd.DataFrame({
            "timestep":            np.arange(T),
            "calories_consumed":   per_step,
            "cumulative_calories": np.cumsum(per_step),
        })

    def plot_calories(self, figsize=(11, 4)):
        """
        Plot cumulative calories consumed over the episode, with sleep windows
        shaded and the end-of-episode total annotated. Inference-only.
        """
        df = self.calorie_summary()
        fig, ax = plt.subplots(figsize=figsize)
        if df.empty:
            print("[FoodEnv] No steps taken yet — nothing to plot.")
            return fig

        x     = df["timestep"].values
        cum   = df["cumulative_calories"].values
        total = float(cum[-1]) if len(cum) else 0.0

        ax.plot(x, cum, color="darkorange", lw=2, label="Cumulative calories")
        ax.fill_between(x, cum, alpha=0.12, color="darkorange")

        for k, (ws, we) in enumerate(self.sleep_windows()):
            ax.axvspan(ws, min(we, x[-1] if len(x) else ws), alpha=0.10,
                       color="navy", label="Sleep" if k == 0 else "")

        if len(x):
            ax.annotate(f"Total: {total:.0f} kcal",
                        xy=(x[-1], total), xytext=(-12, 12),
                        textcoords="offset points", ha="right",
                        color="darkorange", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Time (env steps;  1 step = {self.minutes_per_step} min)")
        ax.set_ylabel("Cumulative calories (kcal)")
        ax.set_title("Total calories consumed over the episode (inference)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def cycle_summary(self) -> pd.DataFrame:
        """Return a DataFrame of the sleep/wake state at each step."""
        if not self._cycle_log:
            return pd.DataFrame(columns=["timestep", "is_awake", "cycle_number"])
        return pd.DataFrame(self._cycle_log)

    def sleep_windows(self):
        """
        Return a list of (start_step, end_step) tuples for every sleep window
        that occurred during the episode.  Useful for shading plots.
        """
        windows = []
        in_sleep  = False
        start     = 0
        for entry in self._cycle_log:
            if not entry["is_awake"] and not in_sleep:
                in_sleep = True
                start    = entry["timestep"]
            elif entry["is_awake"] and in_sleep:
                in_sleep = False
                windows.append((start, entry["timestep"]))
        # Close an open window at the end of the episode
        if in_sleep:
            windows.append((start, self.timepoint))
        return windows

    # ──────────────────────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────────────────────

    def render(self):
        n   = self.num_nutrients
        fig, ax = plt.subplots(figsize=(max(6, n * 2), 3))
        x   = np.arange(n)

        low_err  = self._norm_targets - self._norm_target_low
        high_err = self._norm_target_high - self._norm_targets

        ax.bar(x, self._phys_nutrients, 0.4, label="Agent state", color="steelblue", alpha=0.8)
        ax.errorbar(
            x, self._norm_targets,
            yerr=[low_err, high_err],
            fmt="o", color="crimson", capsize=6, capthick=2, lw=2,
            label="Target range (centre ± tolerance)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(self.nutrient_names, rotation=15, ha="right")
        ax.set_ylabel("Normalised level")
        ax.set_title("Physiological state vs target range")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_consumption(self, max_time: Optional[int] = None, figsize=(12, 8)):
        """
        Plot amount-scaled absorption profiles per nutrient, with sleep windows
        shaded in grey so digestion-during-sleep is visually apparent.
        """
        if not self._ts_nutrient_names:
            print("[FoodEnv] No time-series nutrients active — nothing to plot.")
            return plt.figure()
        if not self._consumption_log:
            print("[FoodEnv] No foods have been consumed yet.")
            return plt.figure()

        T   = max_time if max_time is not None else self.timepoint
        x   = np.arange(T + 1)
        num = len(self._ts_nutrient_names)

        fig, axes = plt.subplots(num, 1, figsize=figsize, sharex=True)
        if num == 1:
            axes = [axes]

        # Pre-compute sleep windows for shading
        windows = self.sleep_windows()

        for n_idx, (ax, n_name) in enumerate(zip(axes, self._ts_nutrient_names)):
            total = np.zeros(T + 1, dtype=np.float64)

            for rec in self._consumption_log:
                profile   = rec["profile"]
                eat_time  = rec["eat_time"]
                food_name = rec["food_name"]
                amount    = rec["amount"]

                contrib = np.zeros(T + 1, dtype=np.float64)
                for age in range(profile.shape[0]):
                    t = eat_time + age
                    if t > T:
                        break
                    contrib[t] += float(profile[age, n_idx]) * amount

                total += contrib
                ax.plot(x, contrib, alpha=0.5, label=f"{food_name} (×{amount:.2f})")

            ax.plot(x, total, color="black", linewidth=2, label="Total")

            # Shade sleep windows in light navy
            for (ws, we) in windows:
                ax.axvspan(ws, min(we, T), alpha=0.10, color="navy", label="Sleep" if ws == windows[0][0] else "")

            ax.set_ylabel(f"{n_name}\n(normalised, scaled)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"Time (env steps;  1 step = {self.minutes_per_step} min)")
        fig.suptitle("Per-nutrient absorption profiles (amount-scaled)", fontsize=14)
        plt.tight_layout()
        return fig