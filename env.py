"""
bio_env.py  –  Biologically-grounded Food RL Environment (Continuous Actions)

Action space  : Box(num_foods,)  — each value in [0, 1]
    amounts[i] = 0.0  → do not eat food at menu slot i
    amounts[i] = 1.0  → eat full portion of food at menu slot i
    amounts[i] = 0.5  → eat half portion, absorption profile scaled by 0.5

Observation space:
    {
        "physiological_state" : Box(state_dim,)           – normalised
        "food_embeddings"     : Box(num_foods, embed_size) – current menu
    }

Nutrient handling is fully driven by the top-level NUTRIENT_CONFIG dict.
Comment/uncomment entries to activate or deactivate nutrients at will.
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
# TOP-LEVEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MINUTES_PER_STEP: int = 2

# Minimum consumption amount to register as "food eaten" for logging purposes.
# Amounts below this threshold are treated as zero (not logged, no absorption).
CONSUMPTION_EPSILON: float = 0.1

NUTRIENT_CONFIG: Dict[str, dict] = {

    "glucose": {
        "csv":            "serum_glucose.csv",
        "col_suffix":     "_serum_glucose_mg_dl",
        "target":         100.0,
        "tolerance":      30.0,
        "in_range_bonus": 0.5,
        "window_size":    1,
        "reward_weight":  1.0,
        "decay_rate":     0.0015,
        "is_cumulative":  False,
    },

    "peptides": {
        "csv":            "small_peptides_absorbed.csv",
        "col_suffix":     "_small peptides absorbed",
        "target":         0.001,
        "tolerance":      0.0005,
        "in_range_bonus": 0.5,
        "window_size":    1,
        "reward_weight":  1.0,
        "decay_rate":     0.005,
        "is_cumulative":  False,
    },

    "fatty_acids": {
        "csv":            "fatty_acids_absorbed.csv",
        "col_suffix":     "_fatty acids absorbed",
        "target":         0.00033,
        "tolerance":      0.00015,
        "in_range_bonus": 0.5,
        "window_size":    1,
        "reward_weight":  1.0,
        "decay_rate":     0.005,
        "is_cumulative":  False,
    },

    # ── HOW TO ADD A NEW NUTRIENT ─────────────────────────────────────────────
    # "my_nutrient": {
    #     "csv":            "my_nutrient.csv",
    #     "col_suffix":     "_my_nutrient_units",
    #     "target":         <float>,
    #     "tolerance":      <float>,
    #     "in_range_bonus": <float>,
    #     "window_size":    <int or None>,
    #     "reward_weight":  <float>,
    #     "decay_rate":     <float>,
    #     "is_cumulative":  <True or False>,
    # },
}


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_normalise(filepath, col_suffix, is_cumulative):
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
    Biologically-grounded food RL environment with CONTINUOUS actions.

    At each step the agent outputs a vector of amounts ∈ [0, 1]^num_foods.
    Absorption of food i is its profile scaled by amounts[i].
    Amounts below CONSUMPTION_EPSILON are treated as zero.

    Parameters
    ──────────
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

        if self.num_nutrients == 0:
            raise ValueError("NUTRIENT_CONFIG is empty — activate at least one nutrient.")

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

        def _norm(val, n):
            rng = max(self._nutrient_maxs[n] - self._nutrient_mins[n], 1e-8)
            return float(np.clip((val - self._nutrient_mins[n]) / rng, 0.0, 1.0))

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

        self.state_dim = self.num_nutrients

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

        # ── Continuous action space: one amount per menu slot ─────────────────
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.num_foods,), dtype=np.float32,
        )

        self._seed = None
        if self.args["seed"] is not None:
            self._set_seed(self.args["seed"])

        self.reset(seed=self.args["seed"])

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
            raise RuntimeError(
                "No food items are common across all active nutrient CSVs."
            )

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

        self._active_foods: List[Dict] = []

        # Full episode consumption log for plotting — never pruned during episode.
        # Each entry: {food_name, eat_time, amount, profile}
        self._consumption_log: List[Dict] = []

        self._cumul_values: Dict[str, float] = {
            n: 0.0 for n in self._cumul_nutrient_names
        }
        self._internal_state: Dict[str, float] = {
            n: 0.0 for n in self._ts_nutrient_names
        }
        self._rolling_buffers: Dict[str, deque] = {}
        for n in self._ts_nutrient_names:
            w = NUTRIENT_CONFIG[n]["window_size"]
            self._rolling_buffers[n] = deque([0.0] * w, maxlen=w)

        self._phys_state = np.zeros(self.state_dim, dtype=np.float32)
        self.timepoint   = 0

        # Per-step consumption summary: list of {timestep, food_name, amount}
        self._step_consumption: List[Dict] = []

        self._sample_menu()
        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, amounts: np.ndarray):
        """
        Execute one environment step with continuous actions.

        Parameters
        ----------
        amounts : (num_foods,) float array, each in [0, 1].
                  amounts[i] scales the absorption profile of menu slot i.
                  Values below CONSUMPTION_EPSILON are treated as zero.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        amounts = np.asarray(amounts, dtype=np.float32)
        assert amounts.shape == (self.num_foods,), (
            f"Expected amounts shape ({self.num_foods},), got {amounts.shape}"
        )

        # ── 1. Register each consumed food ────────────────────────────────────
        for slot_i, amount in enumerate(amounts):
            if float(amount) < CONSUMPTION_EPSILON:
                continue                                    # treat as skip

            food_idx  = int(self._menu[slot_i])
            food_name = self.item_list[food_idx]

            self._active_foods.append({
                "food_idx":  food_idx,
                "food_name": food_name,
                "eat_time":  self.timepoint,
                "amount":    float(amount),                # ← scale factor
            })

            # Full-episode log (for plot_consumption)
            self._consumption_log.append({
                "food_name": food_name,
                "eat_time":  self.timepoint,
                "amount":    float(amount),
                "profile":   self._profiles[food_name],
            })

            # Per-step log (for infer_episode / generate_episode summaries)
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

        # ── 2. Sum AMOUNT-SCALED active absorption profiles ───────────────────
        x_t          = np.zeros(len(self._ts_nutrient_names), dtype=np.float32)
        still_active = []

        for food in self._active_foods:
            age     = self.timepoint - food["eat_time"]
            profile = self._profiles.get(food["food_name"])

            if profile is not None and age < profile.shape[0]:
                # Scale the absorption profile by the consumption amount
                x_t += profile[age] * food["amount"]
                still_active.append(food)

        self._active_foods = still_active

        # ── 3. IS leaky integrator + rolling buffer ───────────────────────────
        phys_state_ts: Dict[str, float] = {}
        for i, n in enumerate(self._ts_nutrient_names):
            dr   = NUTRIENT_CONFIG[n]["decay_rate"]
            is_n = self._internal_state[n] * (1.0 - dr) + float(x_t[i])
            self._internal_state[n] = is_n
            self._rolling_buffers[n].append(is_n)
            phys_state_ts[n] = float(sum(self._rolling_buffers[n]))

        # ── 4. Assemble physiological state ───────────────────────────────────
        state_values = []
        for n in self.nutrient_names:
            if n in phys_state_ts:
                state_values.append(phys_state_ts[n])
            else:
                state_values.append(self._cumul_values.get(n, 0.0))

        self._phys_state = np.array(state_values, dtype=np.float32)

        # ── 5. Compute reward ─────────────────────────────────────────────────
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
            else:
                dist    = state_val - high
                reward += 10 * weight * (-(dist ** 2))

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
        return {
            "physiological_state": self._phys_state.copy(),
            "food_embeddings":     self._all_embeddings[self._menu].copy(),
        }

    def _get_info(self):
        below = np.maximum(0.0, self._norm_target_low  - self._phys_state)
        above = np.maximum(0.0, self._phys_state - self._norm_target_high)
        range_distance = float(np.sum(below + above))

        return {
            "distance":        range_distance,
            "menu":            self._menu.copy(),
            "menu_names":      [self.item_list[i] for i in self._menu],
            "timepoint":       self.timepoint,
            "real_minutes":    self.timepoint * self.minutes_per_step,
            # Per-step consumption detail: list of {food_name, amount} for this step
            "step_consumed":   [
                e for e in self._step_consumption
                if e["timestep"] == self.timepoint - 1
            ],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Summary helpers
    # ──────────────────────────────────────────────────────────────────────────

    def nutrient_norm_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "nutrient":          self.nutrient_names,
            "data_min":          [self._nutrient_mins[n] for n in self.nutrient_names],
            "data_max":          [self._nutrient_maxs[n] for n in self.nutrient_names],
            "raw_target":        [NUTRIENT_CONFIG[n]["target"] for n in self.nutrient_names],
            "normalised_target": list(self._norm_targets),
            "window_size":       [NUTRIENT_CONFIG[n]["window_size"] for n in self.nutrient_names],
            "decay_rate":        [NUTRIENT_CONFIG[n]["decay_rate"] for n in self.nutrient_names],
            "reward_weight":     [NUTRIENT_CONFIG[n]["reward_weight"] for n in self.nutrient_names],
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

    def consumption_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame of every food consumed during the episode,
        with columns [timestep, food_name, amount].
        Useful for post-episode analysis and inference.
        """
        if not self._step_consumption:
            return pd.DataFrame(columns=["timestep", "food_name", "amount"])
        return pd.DataFrame(self._step_consumption)

    # ──────────────────────────────────────────────────────────────────────────
    # Render
    # ──────────────────────────────────────────────────────────────────────────

    def render(self):
        n   = self.state_dim
        fig, ax = plt.subplots(figsize=(max(6, n * 2), 3))
        x   = np.arange(n)

        low_err  = self._norm_targets - self._norm_target_low
        high_err = self._norm_target_high - self._norm_targets

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
        Plot per-step AMOUNT-SCALED absorption contribution of each eaten food
        for every active time-series nutrient.

        The plotted contribution for food i at step t is:
            profile[age, nutrient_idx] * amount_i
        matching exactly what the env adds to x_t during step().
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
                profile   = rec["profile"]      # (T_steps, num_ts_nutrients)
                eat_time  = rec["eat_time"]
                food_name = rec["food_name"]
                amount    = rec["amount"]        # consumption scale factor

                contrib = np.zeros(T + 1, dtype=np.float64)
                for age in range(profile.shape[0]):
                    t = eat_time + age
                    if t > T:
                        break
                    # Scale absorption by the amount consumed
                    contrib[t] += float(profile[age, n_idx]) * amount

                total += contrib
                ax.plot(x, contrib, alpha=0.5, label=f"{food_name} (×{amount:.2f})")

            ax.plot(x, total, color="black", linewidth=2, label="Total")
            ax.set_ylabel(f"{n_name}\n(normalised, scaled)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(f"Time (env steps;  1 step = {self.minutes_per_step} min)")
        fig.suptitle("Per-nutrient absorption profiles (amount-scaled)", fontsize=14)
        plt.tight_layout()
        return fig
