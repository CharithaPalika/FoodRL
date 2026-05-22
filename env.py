"""
bio_env.py  –  Biologically-grounded Food RL Environment

Nutrient handling is fully driven by the top-level NUTRIENT_CONFIG dict.
Comment/uncomment entries to activate or deactivate nutrients at will.
State dimension, rolling windows, decay, and reward weights all adapt
automatically based on which nutrients are active.

Action space  : Discrete(num_foods + 1)
    0           → skip (eat nothing this step)
    1 .. K      → eat the food at menu slot (action - 1)

Observation space:
    {
        "physiological_state" : Box(state_dim,)           – normalised
        "food_embeddings"     : Box(num_foods, embed_size) – current menu
    }
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


# ══════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL CONSTANTS  — edit here, no changes needed inside the class
# ══════════════════════════════════════════════════════════════════════════════

# Each env step covers this many real minutes of absorption data.
MINUTES_PER_STEP: int = 2

# ── Nutrient configuration ────────────────────────────────────────────────────
# To activate a nutrient   : uncomment its block.
# To deactivate a nutrient : comment out its block.
# To add a new nutrient    : copy the placeholder block at the bottom,
#                            fill in your values, and uncomment.
#
# Keys per nutrient:
#   csv           – filename inside food_folder (rows=minutes, cols=foods)
#   col_suffix    – suffix to strip from CSV column names to get food names
#                   e.g. "Biscuit_01_serum_glucose_mg_dl" → strip
#                        "_serum_glucose_mg_dl" → food name = "Biscuit_01"
#   target        – desired physiological level in raw (un-normalised) units
#   window_size   – rolling-average window in env steps (None = cumulative)
#   reward_weight – scalar multiplier for this nutrient's squared-error reward
#   decay_rate    – per-step multiplicative clearance  (0.0 = no decay)
#   is_cumulative – True  : CSV values are running totals; normalise by max
#                   False : CSV values are time-series levels; normalise by L2

NUTRIENT_CONFIG: Dict[str, dict] = {

    "glucose": {
        "csv":           "serum_glucose.csv",
        "col_suffix":    "_serum_glucose_mg_dl",
        "target":        100.0,          # mg/dl  — centre of acceptable range
        "tolerance":     30.0,          # mg/dl  — ± band → range [70, 100] mg/dl
        "in_range_bonus": 0.5,          # positive reward per step when inside range
        "window_size":   1,             # instantaneous
        "reward_weight": 1.0,
        "decay_rate":    0.01, #0.005, #0.05, #0.02,
        "is_cumulative": False,
    },

    "peptides": {
        "csv":           "small_peptides_absorbed.csv",
        "col_suffix":    "_small peptides absorbed",
        "target":        0.001,         # g  — centre of acceptable range
        "tolerance":     0.0005,        # g  — ± band → range [0.0005, 0.0015] g
        "in_range_bonus": 0.5,          # positive reward per step when inside range
        "window_size":   1, #60           # avg over 60 steps = 2 hrs
        "reward_weight": 1.0,
        "decay_rate":    0.005,
        "is_cumulative": False,
    },

    "fatty_acids": {
        "csv":           "fatty_acids_absorbed.csv",
        "col_suffix":    "_fatty acids absorbed",
        "target":        0.00033,       # g  — centre of acceptable range
        "tolerance":     0.00015,       # g  — ± band → range [0.00018, 0.00048] g
        "in_range_bonus": 0.5,          # positive reward per step when inside range
        "window_size":   1,#120,           # avg over 120 steps = 4 hrs
        "reward_weight": 1.0,
        "decay_rate":    0.005,
        "is_cumulative": False,
    },

    # "calories": {
    #     "csv":           "calories_absorbed.csv",
    #     "col_suffix":    "_calories absorbed",
    #     "target":        2200.0,        # kcal/day
    #     "window_size":   None,          # cumulative total — no rolling window
    #     "reward_weight": 0.2,
    #     "decay_rate":    0.0,           # cumulative intake; typically no decay
    #     "is_cumulative": True,
    # },

    # ── HOW TO ADD A NEW NUTRIENT ─────────────────────────────────────────────
    # 1. Place your CSV in food_folder.  Format: rows = minutes, first column
    #    named "time", remaining columns named  "<food_name><col_suffix>".
    # 2. Copy this block, fill in the values, and uncomment.
    #
    # "my_nutrient": {
    #     "csv":           "my_nutrient.csv",
    #     "col_suffix":    "_my_nutrient_units",   # suffix after food name in CSV
    #     "target":        <float>,                # raw target value
    #     "window_size":   <int or None>,          # steps; None = cumulative
    #     "reward_weight": <float>,
    #     "decay_rate":    <float>,                # 0.0 – 1.0
    #     "is_cumulative": <True or False>,
    # },
    # ─────────────────────────────────────────────────────────────────────────
}


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_normalise(
    filepath: str,
    col_suffix: str,
    is_cumulative: bool,
):
    """
    Load a nutrient CSV, strip *col_suffix* from column names to recover food
    names, and normalise.

    Time-series nutrients  (is_cumulative=False)
    ────────────────────────────────────────────
        sum_profile(t) = Σ_i  X(t, i)          (sum across foods at each minute)
        norm           = ||sum_profile||_2
        X̂             = X / norm

    Cumulative nutrients  (is_cumulative=True)
    ─────────────────────────────────────────
        C_i  = last row of column i             (total absorbed over full profile)
        norm = max_i C_i
        Ĉ_i = C_i / norm                       (scalar per food)

    Returns
    ───────
    food_names : list[str]
    norm_data  : pd.DataFrame (T_min × foods)  for time-series
               | pd.Series   (foods,)          for cumulative
    norm       : float  — the normalisation constant
    """
    ext = os.path.splitext(filepath)[1].lower()
    df  = pd.read_excel(filepath) if ext in (".xlsx", ".xls") else pd.read_csv(filepath)

    feature_df = df.drop(columns=["time"]).fillna(0)

    # Strip nutrient suffix from column names → clean food names
    food_names         = [c.replace(col_suffix, "").strip() for c in feature_df.columns]
    feature_df.columns = food_names

    if is_cumulative:
        C_i   = feature_df.iloc[-1].astype(float)   # total absorbed per food
        v_min = float(C_i.min())
        v_max = float(C_i.max())
        denom = (v_max - v_min) if (v_max - v_min) > 0.0 else 1.0
        norm_data = (C_i - v_min) / denom           # Series in [0, 1]
    else:
        v_min = float(feature_df.values.min())
        v_max = float(feature_df.values.max())
        denom = (v_max - v_min) if (v_max - v_min) > 0.0 else 1.0
        norm_data = (feature_df - v_min) / denom    # DataFrame in [0, 1]

    return food_names, norm_data, v_min, v_max


def _build_delta_profile(col: np.ndarray, minutes_per_step: int) -> np.ndarray:
    """
    Convert a normalised cumulative column into signed per-env-step deltas.

    Strides by *minutes_per_step* and takes direct differences:

        Δ̂(s) = X̂(s · M)  −  X̂((s−1) · M)      with  X̂(−1) ≡ 0

    No clipping — deltas are signed so that both absorption (positive)
    and return-to-baseline (negative) are preserved.

    col     : (T_min,)   float32 — normalised cumulative values at 1-min res
    Returns : (T_steps,) float32 — signed delta per env step
    """
    col        = col.astype(np.float32)
    indices    = np.arange(0, len(col), minutes_per_step)   # 0, M, 2M, ...
    subsampled = col[indices]                                # (T_steps,)
    deltas     = np.diff(subsampled, prepend=np.float32(0.0))  # Δ̂(0)=X̂[0], rest are diffs
    return deltas.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# FoodEnv
# ══════════════════════════════════════════════════════════════════════════════

class FoodEnv(gym.Env):
    """
    Biologically-grounded food RL environment.

    Nutrient dynamics, decay, and reward weights are all configured through
    the module-level NUTRIENT_CONFIG dict — no changes required inside this
    class.

    Parameters (passed as keyword arguments)
    ─────────────────────────────────────────
    food_folder       : str   – directory containing the nutrient CSVs
    num_foods         : int   – menu size shown to the agent each step (K)
    max_steps         : int   – episode length
    one_hot_embedding : bool  – use one-hot food embeddings
    embed_size        : int   – learned embedding size (only if not one-hot)
    seed              : int   – RNG seed
    """

    metadata = {"render_modes": []}

    def __init__(self, food_folder: str, **args):
        super().__init__()

        defaults = dict(
            num_foods=5,
            max_steps=50,
            one_hot_embedding=True,
            embed_size=None,
            seed=0,
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
        # print('NUM NUTRIENTS', self.num_nutrients)
        if self.num_nutrients == 0:
            raise ValueError("NUTRIENT_CONFIG is empty — activate at least one nutrient.")

        # ── Load & normalise nutrient data ────────────────────────────────────
        self._nutrient_mins: Dict[str, float] = {}
        self._nutrient_maxs: Dict[str, float] = {}
        self._load_food_library()

        if self.num_foods > self.num_items:
            raise ValueError(
                f"num_foods={self.num_foods} exceeds available food items ({self.num_items})."
            )

        # ── Food embeddings ───────────────────────────────────────────────────
        self.one_hot_embedding = self.args["one_hot_embedding"]
        self.embed_size = (
            self.num_items if self.one_hot_embedding else self.args["embed_size"]
        )
        self._build_food_embeddings()

        # ── Normalised targets and range boundaries ───────────────────────────
        # Centre:  T̂_n     = (target   - min_n) / (max_n - min_n)
        # Low:     T̂_n_low = (target - tol - min_n) / (max_n - min_n)
        # High:    T̂_n_hi  = (target + tol - min_n) / (max_n - min_n)
        # All three are clipped to [0, 1] after normalisation to stay in range.
        def _norm(val, n):
            rng = max(self._nutrient_maxs[n] - self._nutrient_mins[n], 1e-8)
            return float(np.clip((val - self._nutrient_mins[n]) / rng, 0.0, 1.0))

        self._norm_targets = np.array(
            [_norm(NUTRIENT_CONFIG[n]["target"], n) for n in self.nutrient_names],
            dtype=np.float32,
        )
        self._norm_target_low = np.array(
            [
                _norm(NUTRIENT_CONFIG[n]["target"] - NUTRIENT_CONFIG[n].get("tolerance", 0.0), n)
                for n in self.nutrient_names
            ],
            dtype=np.float32,
        )
        self._norm_target_high = np.array(
            [
                _norm(NUTRIENT_CONFIG[n]["target"] + NUTRIENT_CONFIG[n].get("tolerance", 0.0), n)
                for n in self.nutrient_names
            ],
            dtype=np.float32,
        )

        # ── Gym spaces ────────────────────────────────────────────────────────
        # state_dim adapts automatically to however many nutrients are active
        self.state_dim = self.num_nutrients

        self.observation_space = spaces.Dict(
            {
                "physiological_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=np.float32,
                ),
                "food_embeddings": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_foods, self.embed_size),
                    dtype=np.float32,
                ),
            }
        )
        # action=0 → skip;  action=1..K → eat food at menu slot (action-1)
        self.action_space = spaces.Discrete(self.num_foods + 1)

        # ── Seed & initial reset ──────────────────────────────────────────────
        self._seed = None
        if self.args["seed"] is not None:
            self._set_seed(self.args["seed"])

        self.reset(seed=self.args["seed"])

    # ──────────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_food_library(self):
        """
        Load every active nutrient CSV, normalise, intersect food names, and
        build per-food absorption profiles.

        Populates
        ─────────
        self.item_list            : list[str]  – sorted food names
        self.num_items            : int
        self._profiles            : dict[food_name → (T_steps, num_ts_nutrients)]
        self._calorie_scalars     : dict[food_name → float]  (if cumulative active)
        self._ts_nutrient_names   : list[str]  – time-series nutrients (ordered)
        self._cumul_nutrient_names: list[str]  – cumulative nutrients (ordered)
        self._nutrient_norms      : dict[str → float]
        """
        per_nutrient: List[Dict] = []

        for n, cfg in NUTRIENT_CONFIG.items():
            filepath = os.path.join(self.food_folder, cfg["csv"])
            print(f"[FoodEnv] Loading  '{n}'  from  '{filepath}'")

            food_names, norm_data, v_min, v_max = _load_and_normalise(
                filepath,
                cfg["col_suffix"],
                cfg["is_cumulative"],
            )
            self._nutrient_mins[n] = v_min
            self._nutrient_maxs[n] = v_max

            per_nutrient.append(
                {
                    "name":          n,
                    "food_names":    food_names,
                    "data":          norm_data,
                    "is_cumulative": cfg["is_cumulative"],
                }
            )
            print(
                f"           min={v_min:.4f}  max={v_max:.4f}   foods={len(food_names)}"
                + (
                    f"   time_points={norm_data.shape[0]}"
                    if not cfg["is_cumulative"]
                    else ""
                )
            )

        # ── Intersect food names across all active CSVs ────────────────────────
        common = set(per_nutrient[0]["food_names"])
        for entry in per_nutrient[1:]:
            common &= set(entry["food_names"])
        common = sorted(common)

        if not common:
            raise RuntimeError(
                "No food items are common across all active nutrient CSVs.\n"
                "Check that col_suffix entries correctly strip to matching food names."
            )

        dropped = set(per_nutrient[0]["food_names"]) - set(common)
        if dropped:
            print(f"[FoodEnv] WARNING – dropping foods absent from some CSVs: {dropped}")

        self.item_list  = common
        self.num_items  = len(common)

        self._ts_nutrient_names    = [e["name"] for e in per_nutrient if not e["is_cumulative"]]
        self._cumul_nutrient_names = [e["name"] for e in per_nutrient if     e["is_cumulative"]]

        ts_entries    = [e for e in per_nutrient if not e["is_cumulative"]]
        cumul_entries = [e for e in per_nutrient if     e["is_cumulative"]]

        # ── Build per-food (T_steps, num_ts_nutrients) absorption arrays ───────
        self._profiles:         Dict[str, np.ndarray] = {}
        self._calorie_scalars:  Dict[str, Dict[str, float]] = {}

        for food in common:

            # Time-series stack
            if ts_entries:
                cols = []
                for entry in ts_entries:
                    raw_col   = entry["data"][food].values.astype(np.float32)  # (T_min,)
                    step_col  = _build_delta_profile(raw_col, self.minutes_per_step)  # (T_steps,)
                    cols.append(step_col)

                # Pad all nutrients to the same number of steps, then stack
                T_max  = max(len(c) for c in cols)
                padded = np.stack(
                    [
                        np.concatenate(
                            [c, np.zeros(T_max - len(c), dtype=np.float32)]
                        )
                        for c in cols
                    ],
                    axis=1,
                )  # (T_max, num_ts_nutrients)
                self._profiles[food] = padded

            # Cumulative scalars (e.g. total calories)
            scalars = {}
            for entry in cumul_entries:
                scalars[entry["name"]] = float(entry["data"][food])
            self._calorie_scalars[food] = scalars

        print(
            f"[FoodEnv] Ready — {self.num_items} foods | "
            f"{len(self._ts_nutrient_names)} time-series nutrients | "
            f"{len(self._cumul_nutrient_names)} cumulative nutrients"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Seeding
    # ──────────────────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int):
        self._seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ──────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ──────────────────────────────────────────────────────────────────────────

    def _build_food_embeddings(self):
        """Pre-build a (num_items, embed_size) array for fast index lookup."""
        if self.one_hot_embedding:
            self._all_embeddings = np.eye(self.num_items, dtype=np.float32)
        else:
            emb = nn.Embedding(self.num_items, self.embed_size)
            emb.weight.requires_grad_(False)
            self._all_embeddings = emb.weight.detach().cpu().numpy().astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Menu sampling
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_menu(self):
        """Draw num_foods unique food indices without replacement."""
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

        # Active food list: each entry = {food_name, food_idx, eat_time}
        self._active_foods: List[Dict] = []

        # Full episode log for plot_consumption — never pruned during the episode
        self._consumption_log: List[Dict] = []

        # Cumulative totals for non-rolling nutrients (e.g. calories)
        self._cumul_values: Dict[str, float] = {
            n: 0.0 for n in self._cumul_nutrient_names
        }

        # Leaky-integrator internal state — one scalar per time-series nutrient
        self._internal_state: Dict[str, float] = {
            n: 0.0 for n in self._ts_nutrient_names
        }

        # Rolling buffers initialised to zeros for each time-series nutrient
        # Each buffer stores IS values; phys_state = sum(buffer)
        self._rolling_buffers: Dict[str, deque] = {}
        for n in self._ts_nutrient_names:
            w = NUTRIENT_CONFIG[n]["window_size"]
            self._rolling_buffers[n] = deque([0.0] * w, maxlen=w)

        self._phys_state = np.zeros(self.state_dim, dtype=np.float32)
        self.timepoint   = 0

        self._sample_menu()
        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Execute one environment step.

        action = 0          → skip (eat nothing)
        action = 1 .. K     → eat the food at menu slot  (action - 1)
        """

        # ── 1. Eat or skip ────────────────────────────────────────────────────
        if action != 0:
            food_idx  = int(self._menu[action - 1])
            food_name = self.item_list[food_idx]

            self._active_foods.append(
                {
                    "food_idx":  food_idx,
                    "food_name": food_name,
                    "eat_time":  self.timepoint,
                }
            )

            # Log this eating event for plot_consumption (kept for the full episode)
            self._consumption_log.append(
                {
                    "food_name": food_name,
                    "eat_time":  self.timepoint,
                    "profile":   self._profiles[food_name],  # (T_steps, num_ts_nutrients)
                }
            )

            # Add cumulative scalars immediately on eating (e.g. calories)
            for n in self._cumul_nutrient_names:
                self._cumul_values[n] += self._calorie_scalars[food_name].get(n, 0.0)

        # ── 2. Sum active absorption profiles (time-series nutrients) ─────────
        x_t        = np.zeros(len(self._ts_nutrient_names), dtype=np.float32)
        still_active = []

        for food in self._active_foods:
            age     = self.timepoint - food["eat_time"]
            profile = self._profiles.get(food["food_name"])

            if profile is not None and age < profile.shape[0]:
                x_t += profile[age]          # (num_ts_nutrients,)
                still_active.append(food)
            # food whose profile has run its course is simply dropped

        self._active_foods = still_active

        # ── 3. IS leaky integrator + push into rolling buffer ─────────────────
        #
        #   IS_n(t)  =  IS_n(t-1) · (1 - decay_n)  +  ΔX_n(t)
        #   buffer_n  ← IS_n(t)          (window deque, drops oldest)
        #   phys_state_n  =  Σ buffer_n  (window SUM of IS values)
        #
        # Signed IS values are possible (e.g. return-to-baseline dip), so the
        # observation lower bound is -inf.

        phys_state_ts: Dict[str, float] = {}
        for i, n in enumerate(self._ts_nutrient_names):
            dr   = NUTRIENT_CONFIG[n]["decay_rate"]
            # Leaky integrator update
            is_n = self._internal_state[n] * (1.0 - dr) + float(x_t[i])
            self._internal_state[n] = is_n
            # Push IS value into window buffer
            self._rolling_buffers[n].append(is_n)
            # Physiological state = SUM of buffer (not mean)
            phys_state_ts[n] = float(sum(self._rolling_buffers[n]))

        # ── 4. Assemble physiological state in NUTRIENT_CONFIG order ──────────
        state_values = []
        for n in self.nutrient_names:
            if n in phys_state_ts:
                state_values.append(phys_state_ts[n])
            else:
                # Cumulative nutrient (e.g. calories) — no decay, no window
                state_values.append(self._cumul_values.get(n, 0.0))

        self._phys_state = np.array(state_values, dtype=np.float32)

        # ── 6. Compute reward ─────────────────────────────────────────────────
        #
        #   For each nutrient n with acceptable range [low_n, high_n]:
        #
        #     Inside range  → r_n = w_n · in_range_bonus_n          (positive)
        #     Below low     → r_n = w_n · -(low_n  - state_n)²      (quadratic)
        #     Above high    → r_n = w_n · -(state_n - high_n)²      (quadratic)
        #
        reward = 0.0
        for i, n in enumerate(self.nutrient_names):
            cfg       = NUTRIENT_CONFIG[n]
            state_val = float(self._phys_state[i])
            low       = float(self._norm_target_low[i])
            high      = float(self._norm_target_high[i])
            weight    = cfg["reward_weight"]
            bonus     = cfg.get("in_range_bonus", 0.0)

            if low <= state_val <= high:
                reward += weight * bonus
            elif state_val < low:
                dist    = low - state_val
                reward += 10 * weight * (-(dist ** 2))
            else:  # state_val > high
                dist    = state_val - high
                reward += 10 * weight * (-(dist ** 2))

        # ── 7. Bookkeeping ────────────────────────────────────────────────────
        self.timepoint += 1
        terminated      = False
        truncated       = self.timepoint >= self.max_steps
        self._sample_menu()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ──────────────────────────────────────────────────────────────────────────
    # Observation / info
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        return {
            "physiological_state": self._phys_state.copy(),
            "food_embeddings":     self._all_embeddings[self._menu].copy(),
        }

    def _get_info(self):
        # Distance = 0 inside the acceptable range, else distance to nearest boundary.
        # Summed over all nutrients (L1 over range-clipped errors).
        below = np.maximum(0.0, self._norm_target_low  - self._phys_state)
        above = np.maximum(0.0, self._phys_state - self._norm_target_high)
        range_distance = float(np.sum(below + above))

        return {
            "distance":     range_distance,
            "menu":         self._menu.copy(),
            "menu_names":   [self.item_list[i] for i in self._menu],
            "timepoint":    self.timepoint,
            "real_minutes": self.timepoint * self.minutes_per_step,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Nutrient summary helpers
    # ──────────────────────────────────────────────────────────────────────────

    def nutrient_norm_summary(self) -> pd.DataFrame:
        """Return normalisation constants and normalised targets per nutrient."""
        return pd.DataFrame(
            {
                "nutrient":          self.nutrient_names,
                "data_min":          [self._nutrient_mins[n] for n in self.nutrient_names],
                "data_max":          [self._nutrient_maxs[n] for n in self.nutrient_names],
                "raw_target":        [NUTRIENT_CONFIG[n]["target"] for n in self.nutrient_names],
                "normalised_target": list(self._norm_targets),
                "window_size":       [NUTRIENT_CONFIG[n]["window_size"] for n in self.nutrient_names],
                "decay_rate":        [NUTRIENT_CONFIG[n]["decay_rate"] for n in self.nutrient_names],
                "reward_weight":     [NUTRIENT_CONFIG[n]["reward_weight"] for n in self.nutrient_names],
            }
        )

    def food_profile_summary(self) -> pd.DataFrame:
        """Return profile length and total absorption per food per nutrient."""
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

    # ──────────────────────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────────────────────

    def render(self):
        n   = self.state_dim
        fig, ax = plt.subplots(figsize=(max(6, n * 2), 3))
        x   = np.arange(n)

        # Acceptable range shown as error bars around the centre target
        low_err  = self._norm_targets - self._norm_target_low   # downward extent
        high_err = self._norm_target_high - self._norm_targets  # upward extent

        ax.bar(x, self._phys_state, 0.4, label="Agent state", color="steelblue", alpha=0.8)
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
        Plot per-step absorption contribution of each eaten food for every
        active time-series nutrient.  Call after an episode (or mid-episode)
        to see the full digestion history.
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

        for n_idx, (ax, n_name) in enumerate(zip(axes, self._ts_nutrient_names)):
            total = np.zeros(T + 1, dtype=np.float64)

            for rec in self._consumption_log:
                profile   = rec["profile"]          # (T_steps, num_ts_nutrients)
                eat_time  = rec["eat_time"]
                food_name = rec["food_name"]

                # Build a full-length timeline for this food × nutrient
                contrib = np.zeros(T + 1, dtype=np.float64)
                for age in range(profile.shape[0]):
                    t = eat_time + age
                    if t > T:
                        break
                    contrib[t] += float(profile[age, n_idx])

                total += contrib
                ax.plot(x, contrib, alpha=0.5, label=food_name)

            ax.plot(x, total, color="black", linewidth=2, label="Total")
            ax.set_ylabel(f"{n_name}\n(normalised)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"Time (env steps;  1 step = {self.minutes_per_step} min)")
        fig.suptitle("Per-nutrient absorption profiles", fontsize=14)
        plt.tight_layout()
        return fig
