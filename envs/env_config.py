"""
env_config.py – Configuration data for FoodEnv.

Kept separate from env.py so nutrient definitions (and other config constants)
can be edited/versioned without touching environment logic.
"""


from typing import Dict


# ── Nutrient configuration ────────────────────────────────────────────────────
# New sleep-specific keys vs the original config:
#   sleep_decay_rate     : decay rate applied while the agent is asleep.
#                          Nutrients clear more slowly during sleep — the body
#                          is not actively metabolising food at the same rate.
#   sleep_penalty_weight : multiplier applied on top of reward_weight when the
#                          nutrient falls BELOW its low boundary during sleep.
#                          Being depleted while asleep is worse because the
#                          agent cannot eat to recover until waking.
#
# OPTIONAL initial condition:
#   initial_value        : level the nutrient STARTS each episode at, in the
#                          SAME raw units as `target`. If omitted (default),
#                          the nutrient starts at an empty integrator (phys
#                          state 0.0) — the original behaviour. Set it slightly
#                          below `target` to start the agent just under its
#                          band instead of from empty. Handled window-correctly
#                          in env.py, so it works for any window_size.


NUTRIENT_CONFIG: Dict[str, dict] = {

    "glucose": {
        "csv":                "serum_glucose.csv",
        "col_suffix":         "_serum_glucose_mg_dl",
        "target":             90.0, #80.0,       # mg/dl  — centre of acceptable range
        "tolerance":          20.0,        # mg/dl  — ± band → [70, 130] mg/dl
        "initial_value":    60.0,        # mg/dl — start just below target; omit → start empty (0.0)
        "in_range_bonus":     2.0, #0.5,         # reward per step while inside range
        "window_size":        1,           # instantaneous (no rolling average)
        "reward_weight":      0.1, #1.0,
        "decay_rate":         0.02, #0.01, #0.01,        # awake clearance per step
        "sleep_decay_rate":   0.00050,#0.000075,#0.003,       # slower clearance during sleep
        "sleep_penalty_weight": 2.0,       # 2× penalty for depleted glucose at night
        "is_cumulative":      False,
    },

    "peptides": {
        "csv":                "small_peptides_absorbed.csv",
        "col_suffix":         "_small peptides absorbed",
        "target":             0.0004,#0.0003, #0.001,       # g
        "tolerance":          0.00025, #0.0005,      # g  → [0.0005, 0.0015] g
        "initial_value":    0.0002,#0.0003, #0.001,
        "in_range_bonus":     1.5, #0.5,
        "window_size":        1,
        "reward_weight":      0.1, #1.0,
        "decay_rate":         0.02, #0.01, #0.005,
        "sleep_decay_rate":   1e-4,#1e-8,#0.000001,#0.001,
        "sleep_penalty_weight": 1.5,
        "is_cumulative":      False,
    },

    "fatty_acids": {
        "csv":                "fatty_acids_absorbed.csv",
        "col_suffix":         "_fatty acids absorbed",
        "target":             0.00025,#0.00015,#0.00033,     # g
        "tolerance":          0.000075,#0.00015,     # g  → [0.00018, 0.00048] g
        "initial_value":    0.0001,#0.00015,#0.00033,
        "in_range_bonus":     1.5, #0.5,
        "window_size":        1,
        "reward_weight":      0.1, #1.0,
        "decay_rate":         0.02, #0.01, #0.005,
        "sleep_decay_rate":   1e-4,#0.0, #0.00001,#0.001,
        "sleep_penalty_weight": 1.5,
        "is_cumulative":      False,
    },

    # ── HOW TO ADD A NEW NUTRIENT ─────────────────────────────────────────────
    # "my_nutrient": {
    #     "csv":                "my_nutrient.csv",
    #     "col_suffix":         "_my_nutrient_units",
    #     "target":             <float>,
    #     "tolerance":          <float>,
    #     "in_range_bonus":     <float>,
    #     "window_size":        <int or None>,
    #     "reward_weight":      <float>,
    #     "decay_rate":         <float>,
    #     "sleep_decay_rate":   <float>,       # ← required for sleep support
    #     "sleep_penalty_weight": <float>,     # ← required for sleep support
    #     "is_cumulative":      <True or False>,
    #     # "initial_value":    <float>,       # ← OPTIONAL, raw units; omit → start at 0.0
    # },
}


# ── Shadow nutrient configuration ─────────────────────────────────────────────
# Deliberately a SEPARATE dict from NUTRIENT_CONFIG. Entries here are loaded
# and integrated by FoodEnv using the exact same leaky-integrator machinery as
# the real nutrients (same _active_foods, same is_awake gating, same
# _load_and_normalise / _build_delta_profile pipeline) — but the result is
# NEVER added to the observation space and NEVER touches the reward. They
# exist purely so post-hoc analysis (latent_analysis.py) can ask "does the
# learned bottleneck z correlate with / decode fullness or hunger?" without
# fullness/hunger having been available to the policy during training.
#
# Because nothing here is regulated, there is no target/tolerance/reward_weight
# /in_range_bonus/window_size — those only make sense for a nutrient the agent
# is trying to keep in-range. Shadow nutrients are observed-only.
#
# decay_rate / sleep_decay_rate below are PLACEHOLDERS — there's no principled
# value derivable from the fullness.csv / hunger.csv data alone (each food's
# own curve already encodes its own rise/decay shape over ~500 raw minutes).
# Treat these as a starting point to tune, not a derived constant. Set both
# to 0.0 if you want pure leaky accumulation with no extra forgetting beyond
# what's already baked into each food's profile.
#
# OPTIONAL initial condition:
#   initial_value : level the shadow nutrient STARTS each episode at. Omit
#                   (default) → starts at 0.0. NOTE the scale differs from the
#                   real nutrients: a shadow nutrient has no target / raw units,
#                   so initial_value here is on the INTEGRATOR scale directly
#                   (typically ~0–1), used as-is with no normalisation. E.g. set
#                   hunger's initial_value to 0.5 so the agent doesn't begin an
#                   episode at zero hunger.


SHADOW_NUTRIENT_CONFIG: Dict[str, dict] = {

    "fullness": {
        "csv":                "fullness.csv",
        "col_suffix":         "_fullness",
        "decay_rate":         0.02,        # PLACEHOLDER — tune me
        "sleep_decay_rate":   0.0,        # PLACEHOLDER — tune me,
        "initial_value": 0.0,
        "is_cumulative":      False,
    },

    "hunger": {
        "csv":                "hunger.csv",
        "col_suffix":         "_hunger",
        "decay_rate":         0.02,        # PLACEHOLDER — tune me
        "sleep_decay_rate":   0.0,        # PLACEHOLDER — tune me
        "initial_value":    0.5,        # integrator scale; omit → start at 0.0
        "is_cumulative":      False,
    },

    # ── Gut/appetite hormones — same convention as fullness/hunger above:
    # loaded and integrated with the identical leaky-integrator machinery,
    # never touching observation/reward. decay_rate / sleep_decay_rate are
    # PLACEHOLDERS (same caveat as fullness/hunger's — no principled value
    # derivable from the raw CSV alone); tune once you have a sense of how
    # fast each hormone should clear in this env's step size.
    "cck": {
        "csv":                "cck.csv",
        "col_suffix":         "_CCK",
        "decay_rate":         0.01,        # PLACEHOLDER — tune me
        "sleep_decay_rate":   0.0,        # PLACEHOLDER — tune me
        "initial_value": 0.0,
        "is_cumulative":      False,
    },

    "ghrelin": {
        "csv":                "ghrelin.csv",
        "col_suffix":         "_Ghrelin",
        "decay_rate":         0.01,        # PLACEHOLDER — tune me
        "sleep_decay_rate":   0.0,        # PLACEHOLDER — tune me
        "initial_value": 0.5,
        "is_cumulative":      False,
    },

    "glp_1": {
        "csv":                "glp_1.csv",
        "col_suffix":         "_GLP1",
        "decay_rate":         0.01,        # PLACEHOLDER — tune me
        "sleep_decay_rate":   0.0,        # PLACEHOLDER — tune me
        "initial_value": 0.0,
        "is_cumulative":      False,
    },

    "pyy": {
        "csv":                "pyy.csv",
        "col_suffix":         "_PYY",
        "decay_rate":         0.01,        # PLACEHOLDER — tune me
        "sleep_decay_rate":   0.0,        # PLACEHOLDER — tune me
        "initial_value": 0.0,
        "is_cumulative":      False,
    },

    # ── HOW TO ADD A NEW SHADOW NUTRIENT ───────────────────────────────────
    # "my_signal": {
    #     "csv":              "my_signal.csv",
    #     "col_suffix":       "_my_signal",
    #     "decay_rate":       <float>,
    #     "sleep_decay_rate": <float>,
    #     "is_cumulative":    <True or False>,
    #     # "initial_value":  <float>,       # ← OPTIONAL, integrator scale; omit → start at 0.0
    # },
}


# ── Discrete action-space configuration ───────────────────────────────────────
# Only used when FoodEnv(..., is_continuous=False).
#
# Discrete action semantics: Discrete(num_foods + 1)
#   action == 0        → eat nothing this step
#   action == k (k>=1) → eat from menu slot (k-1), at a FIXED fraction
#                        (no portion control in discrete mode — at most one
#                        slot is eaten per step, always at this amount).

DISCRETE_EAT_AMOUNT: float = 0.6 #0.25