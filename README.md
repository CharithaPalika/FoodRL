> **Paper:** *Faulty States, Faulty Learning: Distinct Routes to Dysregulated Feeding in Homeostatic Reinforcement Learning* — C. Palika, A. Khan, R. Maradana, V. Srinivasa Chakravarthy (IIT Madras). Submitted to journal.

# FoodRL 🍽️🧠

FoodRL is the research codebase accompanying the paper *"Faulty States, Faulty Learning: Distinct Routes to Dysregulated Feeding in Homeostatic Reinforcement Learning"*.

Homeostatic reinforcement learning offers a framework for adaptive feeding through regulation of internal physiological states, but the computational origins of dysregulated feeding remain poorly understood. This repo implements that framework: an RL agent learns adaptive feeding from physiological consequences in a partially observable environment with delayed, overlapping nutrient dynamics. During learning, a low-dimensional physiological representation emerges inside the network that encodes hunger- and fullness-related information despite receiving no direct supervision for either signal — an interoception-like internal state for feeding control.

The experiments then disrupt this learned regulation at two distinct levels:

- **Representation level (faulty states):** perturbing the learned interoceptive bottleneck makes the agent act on systematically distorted physiological information, producing under- or over-eating. See `latent_ablations.ipynb`.
- **Learning level (faulty learning):** selectively attenuating positive or negative TD errors during retraining leaves the physiological representation intact but produces distinct dysregulated feeding phenotypes. See `td_clip_retrain*.ipynb`.

Together, these show that similar behavioral phenotypes can arise from distinct computational failures — how physiological state is *represented* vs. how action consequences are *learned*.

This is research/simulation code, not medical or nutrition advice.

## What The Agent Learns 🎯

At each environment step, the agent receives:

- a physiological vector with regulated nutrients plus sleep/wake timing
- embeddings for the currently available food menu

It then chooses either:

- continuous food amounts in `[0, 1]` for each menu slot, or
- one discrete food choice plus a fixed portion amount

The environment integrates nutrient absorption curves over time, applies wake/sleep-specific decay, and rewards the policy for staying near configured target ranges.

Currently configured regulated nutrients live in [`envs/env_config.py`](envs/env_config.py):

- glucose
- peptides
- fatty acids

The repo also tracks "shadow" appetite signals such as fullness, hunger, CCK, ghrelin, GLP-1, and PYY. These are loaded and logged for analysis, but they are intentionally hidden from the policy and do not affect reward. That makes them useful for asking whether the learned bottleneck represents appetite-related signals it never directly observed.

## Repository Map 🗺️

```text
agents/ppo.py             PPO agent, training loop, checkpointing, TD-error logging
envs/env.py               Gymnasium FoodEnv: menus, digestion, rewards, sleep/wake cycles
envs/env_config.py        Nutrient targets, decay rates, shadow signals, portion size
models/model.py           Actor/critic networks with bottleneck modules
analysis/latent_analysis.py  PCA/t-SNE, bottleneck capture, shadow-signal decoding, ablations
utils/                    Observation helpers, dataset pruning/scaling
food_dataset/             Per-food time-series response CSVs (nutrients + appetite signals)
notebooks (top level)     Experimental entry points (see below)
results/                  Checkpoints, training logs, and experiment outputs
plots/                    Final exported figure panels (fig1–fig5), generated output
wandb/, temp/             Run tracking and scratch space — safe to ignore
```

The `food_dataset/` CSVs contain time-series responses for each food. `FoodEnv` uses the active nutrient files (glucose, peptides, fatty acids) for training and loads `calories_absorbed.csv` only for inference-time calorie reporting. "Shadow" appetite signals (fullness, hunger, CCK, ghrelin, GLP-1, PYY) are logged for analysis but hidden from the policy and excluded from reward — useful for asking whether the learned bottleneck represents signals it never directly observed.

### Notebooks 📓

The notebooks are the main experimental entry points; the Python modules provide the reusable code. Run them in roughly this order to reproduce the paper:

1. **`run_training.ipynb`** — Train baseline PPO agents on `FoodEnv`. Configures the model (bottleneck layout, action space), runs training over seeds, and saves checkpoints plus per-episode logs to `results/training_checkpoints/` and `results/training_data/`.

2. **`infer_run.ipynb`** — Load trained checkpoints and run inference. Replays episodes, records per-step/per-episode behavior, and captures bottleneck activations to test whether the latent state decodes hunger/fullness and other shadow signals. Outputs go to `results/plots_inference_normal/`.

3. **`latent_ablations.ipynb`** — The "faulty states" experiments. Perturbs specific bottleneck dimensions (via amplitude scaling and bias shifts, without touching saved weights) to distort the agent's interoceptive representation, then measures the resulting under-/over-eating. Also includes per-neuron probing. Outputs go to `results/plots_latent_ablation/` and `results/plots_neuron_ablation/`.

4. **`td_clip_retrain.ipynb`** — The "faulty learning" experiments. Retrains agents with positive or negative TD errors selectively clipped at a chosen limit, leaving the representation intact while altering how consequences are learned.

5. **`td_clip_retrain_sweep.ipynb`** — Sweeps the TD-clip conditions across multiple seeds and severities; saves runs to `results/retrain_500ep/`.

6. **`td_clip_retrain_plots.ipynb`** — Aggregates the sweep outputs into summary curves and severity plots (`results/plots_td_clip_retrain/`, `results/plots_retrain/`).

7. **`plotting_main.ipynb`** — Builds the final paper figure panels (fig1–fig5) from all saved artifacts into `plots/`.

## Quick Start 🚀

Install dependencies:

```bash
pip install -r requirements.txt
```

[`requirements.txt`](requirements.txt) covers the core stack (unpinned versions):

- **Core scientific:** `numpy`, `pandas`, `scipy`
- **Deep learning / RL:** `torch`, `gymnasium`
- **Analysis utilities:** `scikit-learn`, `tqdm`
- **Plotting:** `matplotlib`
- **Notebooks:** `jupyter`, `ipykernel`

Optional experiment tracking:

```bash
pip install wandb
```

Minimal environment smoke test:

```python
from envs.env import FoodEnv

env = FoodEnv("food_dataset", num_foods=5, max_steps=10, seed=0)
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(obs["physiological_state"].shape)
print(reward)
```

For a full run, start with:

1. `run_training.ipynb` to train baseline agents
2. `infer_run.ipynb` to evaluate trained policies
3. `latent_ablations.ipynb` to probe learned bottlenecks
4. `td_clip_retrain_sweep.ipynb` and `td_clip_retrain_plots.ipynb` for TD-clip experiments
5. `plotting_main.ipynb` to regenerate paper-style figures

## How The Pieces Fit Together 🔁

1. `FoodEnv` loads the food response curves from `food_dataset/`.
2. `PPOAgent` builds an actor/critic model from `models/model.py`.
3. During training, the agent samples menus, chooses foods, and receives reward from nutrient regulation.
4. Checkpoints and per-episode logs are saved under `results/`.
5. Inference notebooks replay trained agents and save per-step/per-episode summaries.
6. Latent-analysis code reads bottleneck activations and tests whether they encode nutrient deficits, appetite signals, and future actions.
7. Plotting notebooks turn saved artifacts into the final figures.

## Bottleneck Experiments 🔬

The model code supports several bottleneck layouts:

- shared actor-critic trunk bottlenecks
- separate actor and critic bottlenecks
- separate physiological and food branches
- optional actor/critic head bottlenecks

`PPOAgent` exposes helpers such as `list_bottlenecks()`, `set_amplitude()`, and `set_bias()` so notebooks can perturb specific latent dimensions without modifying saved model weights. This is what powers the latent ablation and neuron-probing experiments.


