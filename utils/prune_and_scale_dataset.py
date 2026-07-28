"""
prune_and_scale_dataset.py — build a pruned, portion-scaled copy of the food dataset.

Two independent transforms, applied in order:

  1. PRUNE   Drop food columns that are unusable (CRASHED_FOODS) or redundant
             (DUPLICATE_FOODS).
  2. SCALE   Rescale every remaining nutrient value to the level the agent would
             reach after eating a fraction EAT_AMOUNT of a full portion, using
             the SAME arithmetic FoodEnv applies internally.

Why the scaling is affine (and anchored at the resting baseline)
────────────────────────────────────────────────────────────────
A partial portion should not simply multiply the raw curve. Each nutrient has a
pre-meal RESTING level B (fasting glucose ~85 mg/dL, resting ghrelin ~14,
resting hunger ~7.6, absorbed nutrients ~0). Eating scales only the EXCURSION
away from that resting level, not the resting level itself:

    raw_new = amount * (raw - B) + B                                # <- affine

A plain ``amount * raw`` (equivalently, anchoring at 0) collapses the baseline —
e.g. a 0.25 glucose curve would fall to ~17-35 mg/dL instead of staying near
fasting ~85. This is correct for BOTH response directions:

    * rise-from-baseline nutrients (glucose, CCK, fullness, PYY, GLP-1,
      fatty_acids, peptides, calories) rise LESS above B for a smaller portion;
    * drop-from-baseline hormones (ghrelin, hunger) fall LESS below B.

Why B, not the env's normalisation min v_min
─────────────────────────────────────────────
FoodEnv's internal amount-scaling is equivalent to anchoring at v_min (the
global min used by ``_load_and_normalise``): with decay off,

    state_norm = amount * (raw - v_min) / (v_max - v_min)
    -> raw_equiv = amount * (raw - v_min) + v_min

That is faithful to the env but physiologically wrong wherever v_min != the
resting level: for glucose v_min ~= 70 sits BELOW fasting (an insulin
undershoot), and for the drop-hormones v_min is the post-meal NADIR, so anchoring
there scales the fasting level itself. Anchoring at the resting baseline B fixes
this for every nutrient. B is taken per output file as the mean of the t=0
(first-row) values across the pruned foods — the shared 'general food level'
every food starts from. (For the absorbed nutrients B ~= 0, so this coincides
with plain scaling.)

NOTE: because B differs from v_min, this dataset is DELIBERATELY not identical to
running FoodEnv with amount=0.25 internally — it is the physiologically-corrected
partial-portion trajectory in raw units (mg/dL, g, kcal), intended for analysis
and for a corrected regulation target range.

Usage
─────
Run from the repository root as a module so the `envs.env_config` import
resolves against the package layout:

    python -m utils.prune_and_scale_dataset
    python -m utils.prune_and_scale_dataset --amount 0.5 --dst foods_dataset_0p5

    # pruned but unscaled — this is the folder to point FoodEnv at for retraining
    python -m utils.prune_and_scale_dataset --amount 1.0 --dst foods_dataset_pruned
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from envs.env_config import (
    DISCRETE_EAT_AMOUNT,
    NUTRIENT_CONFIG,
    SHADOW_NUTRIENT_CONFIG,
)

# ── Configuration ─────────────────────────────────────────────────────────────

# Source is the RAW, full 28-food dataset. After the repo reorg the pruned +
# scaled output already lives at ./foods_dataset, so the untouched original is
# kept under OLD_FILES/. Override with --src if yours is elsewhere.
DEFAULT_SRC = "OLD_FILES/foods_dataset"
DEFAULT_DST = "foods_dataset"

# The time axis is an index, never a measurement — never scaled, never dropped.
TIME_COLUMN = "time"

# calories_absorbed.csv is not part of NUTRIENT_CONFIG or SHADOW_NUTRIENT_CONFIG:
# FoodEnv loads it separately in _load_calorie_totals() for inference-only kcal
# reporting. It still needs pruning and scaling, so it is declared here.
# Keep in sync with CALORIE_CSV / CALORIE_COL_SUFFIX in env.py.
CALORIE_FILE = "calories_absorbed.csv"
CALORIE_COL_SUFFIX = "_calories absorbed"

# Foods dropped because their simulations are unusable across all 10 files.
#
# Group A — numerical blow-up. CCK and PYY diverge exponentially and serum
# glucose collapses to non-physiological levels before the solver emits NaN:
#   Rapid_starch_01  dies t=111, PYY 7207 (35x the highest healthy food)
#   Glucose_02       dies t=47,  PYY 3023, glucose down to 56.6 mg/dL
#   Biscuit_01       dies t=91,  PYY 1122
#   Glucose_01       dies t=19,  peaks look normal but CCK is doubling every
#                                step (0.98 -> 3.63 -> 9.30 -> 19.90 -> 37.03)
#
# Group B — food properties never reached the simulation. Ghee (pure fat) and
# Eggwhite (pure protein) produce byte-identical output in 9 of 10 channels,
# fatty-acid absorption is 0.0 for ghee, and serum glucose never leaves its
# 84.933 baseline. All three die at exactly t=71:
#   Chickenbreast_01, Eggwhite_01, Ghee_01
#
# These 7 foods also hold 100% of the dataset's NaNs (3019 per file), so the
# pruned output is NaN-free and FoodEnv's fillna(0) never fires.
CRASHED_FOODS: Tuple[str, ...] = (
    "Rapid_starch_01",
    "Glucose_01",
    "Glucose_02",
    "Biscuit_01",
    "Chickenbreast_01",
    "Eggwhite_01",
    "Ghee_01",
)

# Foods dropped because they duplicate another column bit-for-bit.
#
# Muttonpulao_01 is byte-identical to Lemonrice_01 in all 10 files (correlation
# exactly 1.000000 on every channel), so one label is holding the other's data.
#
# Lemonrice_01 is the one kept. Muttonpulao_01 shows no evidence of being a
# genuine mutton pulao trace: ranked by mean correlation against the real
# Muttonpulao_02, it places third (0.904) behind Cholebhature_01 (0.964) and
# Vegetable_Biryani_01 (0.947) — i.e. two unrelated dishes resemble MP02 more
# than "MP01" does. Keeping Lemonrice_01 preserves the identical trajectory
# while leaving a coherent menu (one lemon rice, one mutton pulao via MP02)
# rather than two mutton pulao entries where one contains lemon rice data.
#
# Dropping this matters for training: a duplicated item doubles that food's
# probability of appearing in the sampled menu.
DUPLICATE_FOODS: Tuple[str, ...] = (
    "Muttonpulao_01",
)


# ── Column-suffix registry ────────────────────────────────────────────────────

def build_suffix_map() -> Dict[str, str]:
    """
    Map ``filename -> column suffix`` for every CSV the env knows about.

    Derived from env_config so the two cannot drift apart; the calorie file is
    appended manually because it lives outside both nutrient dicts.
    """
    suffixes = {
        cfg["csv"]: cfg["col_suffix"]
        for cfg in (*NUTRIENT_CONFIG.values(), *SHADOW_NUTRIENT_CONFIG.values())
    }
    suffixes[CALORIE_FILE] = CALORIE_COL_SUFFIX
    return suffixes


def food_from_column(column: str, suffix: str) -> str:
    """Strip a nutrient suffix off a column header to recover the food name."""
    return re.sub(f"{re.escape(suffix)}$", "", column).strip()


# ── Core transforms ───────────────────────────────────────────────────────────

def prune_columns(
    df: pd.DataFrame,
    suffix: str,
    drop_foods: Iterable[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop the columns belonging to `drop_foods`, preserving original column order.

    Returns the pruned frame and the food names actually dropped, so the caller
    can detect a name in `drop_foods` that matched nothing (typo, or a food that
    exists in some files but not others).
    """
    drop_set = set(drop_foods)
    dropped: List[str] = []
    keep: List[str] = [TIME_COLUMN]

    for column in df.columns:
        if column == TIME_COLUMN:
            continue
        food = food_from_column(column, suffix)
        if food in drop_set:
            dropped.append(food)
        else:
            keep.append(column)

    return df[keep], dropped


def resting_baseline(df: pd.DataFrame, columns: List[str]) -> float:
    """The nutrient's pre-meal resting level: mean of the t=0 (first) row across
    all food columns. Every food's curve starts from the same fasting/resting
    physiological state before digestion, so this is the shared 'general food
    level' the partial-portion scaling should pivot around."""
    return float(df[columns].fillna(0.0).iloc[0].mean())


def scale_values(df: pd.DataFrame, amount: float) -> pd.DataFrame:
    """Rescale nutrient columns to a partial portion, anchored at the resting
    baseline.

    Applies ``new = amount * (raw - B) + B`` to every nutrient column, where
    ``B`` is the pre-meal resting level (t=0, averaged over foods; see
    ``resting_baseline``). Eating a fraction ``amount`` of a portion produces a
    fraction of the excursion away from ``B`` while the resting level itself is
    held fixed. This is correct for BOTH directions of response:

      * rise-from-baseline nutrients (glucose, CCK, fullness, PYY, GLP-1,
        fatty_acids, peptides, calories) stay pinned at ``B`` and rise less;
      * drop-from-baseline hormones (ghrelin, hunger) stay pinned at ``B`` and
        fall less — a smaller meal suppresses them less.

    ``B`` is used INSTEAD of the env's normalisation min ``v_min`` (see module
    docstring): ``v_min`` is the global minimum, which for glucose sits below
    fasting and for the drop-hormones is the post-meal nadir — anchoring there
    scales the fasting level itself and is physiologically wrong. The time axis
    is left untouched.
    """
    scaled = df.copy()
    nutrient_columns = [c for c in scaled.columns if c != TIME_COLUMN]
    block = scaled[nutrient_columns].fillna(0.0)
    baseline = resting_baseline(df, nutrient_columns)     # t=0 resting level
    scaled[nutrient_columns] = amount * (block - baseline) + baseline
    return scaled


def transform_file(
    src_path: str,
    suffix: str,
    drop_foods: Iterable[str],
    amount: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load one CSV, prune its food columns, and scale the remainder."""
    df = pd.read_csv(src_path)
    pruned, dropped = prune_columns(df, suffix, drop_foods)
    return scale_values(pruned, amount), dropped


# ── Driver ────────────────────────────────────────────────────────────────────

def build_dataset(
    src_dir: str,
    dst_dir: str,
    drop_foods: Sequence[str],
    amount: float,
) -> None:
    """Transform every known CSV in `src_dir` and write the result to `dst_dir`."""
    suffix_map = build_suffix_map()
    os.makedirs(dst_dir, exist_ok=True)

    print(f"[prune] source      : {src_dir}")
    print(f"[prune] destination : {dst_dir}")
    print(f"[prune] eat amount  : {amount}")
    print(f"[prune] dropping    : {len(drop_foods)} food(s)\n")

    for filename, suffix in sorted(suffix_map.items()):
        src_path = os.path.join(src_dir, filename)
        if not os.path.exists(src_path):
            print(f"  SKIP  {filename:<30} (not found in {src_dir})")
            continue

        result, dropped = transform_file(src_path, suffix, drop_foods, amount)
        result.to_csv(os.path.join(dst_dir, filename), index=False)

        missed = set(drop_foods) - set(dropped)
        note = f"  !! no column matched: {sorted(missed)}" if missed else ""
        n_foods = result.shape[1] - 1
        n_nan = int(result.isna().sum().sum())
        print(
            f"  OK    {filename:<30} foods={n_foods:>3}  rows={len(result):>4}  "
            f"nan={n_nan}{note}"
        )

    print(f"\n[prune] done — {len(suffix_map)} file(s) processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune unusable foods and scale the dataset to a partial portion."
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="source dataset folder")
    parser.add_argument("--dst", default=DEFAULT_DST, help="output folder")
    parser.add_argument(
        "--amount",
        type=float,
        default=DISCRETE_EAT_AMOUNT,
        help="portion fraction applied to every nutrient value "
             f"(default: DISCRETE_EAT_AMOUNT = {DISCRETE_EAT_AMOUNT})",
    )
    parser.add_argument(
        "--keep-crashed",
        action="store_true",
        help="do not drop the crashed foods",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="do not drop the duplicated foods",
    )
    return parser.parse_args()


def resolve_drop_list(keep_crashed: bool, keep_duplicates: bool) -> Tuple[str, ...]:
    """Assemble the set of foods to drop from the enabled exclusion groups."""
    drop: Tuple[str, ...] = ()
    if not keep_crashed:
        drop += CRASHED_FOODS
    if not keep_duplicates:
        drop += DUPLICATE_FOODS
    return drop


def main() -> None:
    args = parse_args()
    drop = resolve_drop_list(args.keep_crashed, args.keep_duplicates)
    build_dataset(args.src, args.dst, drop, args.amount)


if __name__ == "__main__":
    main()
