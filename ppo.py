import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import warnings
import copy
from collections import deque
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from model import (
    SharedActorCritic, Actor, Critic,
    sample_continuous_action, recompute_log_prob, _split_mu_logstd,
)
from utils import obs_to_tensors
from env import FoodEnv


# ──────────────────────────────────────────────────────────────────────────────
# PPO Agent  —  Continuous action space
#
# Action:  amounts ∈ [0, 1]^num_foods  (one consumption fraction per menu slot)
# Policy:  Clipped Gaussian — network predicts (mu, log_std) per food,
#          samples x_raw ~ N(mu, std), clips to [0,1] for the environment.
#          Log-prob computed on x_raw (unclipped) following standard PPO practice.
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
        )

        unknown = set(kwargs) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown init args: {unknown}")

        self.args   = {**defaults, **kwargs}
        self.device = device
        self.env    = env

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
        seed = self.args["seed"]
        if self.args["shared"]:
            self.policy = (
                self.args["shared_ac_network"]
                if self.args["shared_ac_network"] is not None
                else SharedActorCritic(
                    state_size=state_shape,
                    food_flat_size=food_flat_size,
                    num_foods=num_foods,
                    seed=seed,
                )
            ).to(self.device)
            self.actor  = None
            self.critic = None
        else:
            self.actor = (
                self.args["actor_network"]
                if self.args["actor_network"] is not None
                else Actor(
                    state_size=state_shape,
                    food_flat_size=food_flat_size,
                    num_foods=num_foods,
                    seed=seed,
                )
            ).to(self.device)
            self.critic = (
                self.args["critic_network"]
                if self.args["critic_network"] is not None
                else Critic(
                    state_size=state_shape,
                    food_flat_size=food_flat_size,
                    seed=seed,
                )
            ).to(self.device)
            self.policy = None

        self.policy_optimizer = None
        self.actor_optimizer  = None
        self.critic_optimizer = None

        self.episode_returns    = []
        self.episode_consumption = []   # replaces food_eaten; tracks sum of amounts per episode

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

    def _compute_gae(self, rewards, values, dones):
        advs = np.zeros_like(rewards)
        gae  = 0.0
        for t in reversed(range(len(rewards))):
            delta   = rewards[t] + self.args["gamma"] * values[t + 1] * (1 - dones[t]) - values[t]
            delta  *= self.args["limit_delta"]
            gae     = delta + self.args["gamma"] * self.args["lam"] * (1 - dones[t]) * gae
            advs[t] = gae
        returns = advs + values[:-1]
        return advs, returns

    # ──────────────────────────────────────────────────────────────────────────
    # Act
    # ──────────────────────────────────────────────────────────────────────────

    def act(self, obs, deterministic=False):
        """
        Returns
        -------
        amounts   : (num_foods,) tensor  in [0, 1]   — clipped for environment
        log_prob  : scalar tensor                    — sum log-prob over foods
        value     : scalar tensor
        mu        : (num_foods,) tensor              — Gaussian mean
        log_std   : (num_foods,) tensor              — Gaussian log std
        x_raw     : (num_foods,) tensor              — unclipped sample (store for PPO update)
        """
        phy, food_flat = obs_to_tensors(obs, self.device)

        if self.args["shared"]:
            mu, log_std, value = self.policy(phy, food_flat)
        else:
            mu, log_std = self.actor(phy, food_flat)
            value       = self.critic(phy, food_flat)

        if deterministic:
            # At inference: use the mean directly, clipped to [0,1]
            amounts  = torch.clamp(mu, 0.0, 1.0)
            log_prob = None
            x_raw    = mu
        else:
            amounts, log_prob, _, x_raw = sample_continuous_action(mu, log_std)

        return amounts, log_prob, value, mu, log_std, x_raw

    # ──────────────────────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, log_wandb=True, printing=True, **kwargs):

        train_defaults = dict(
            num_episodes=10_000,
            shared_ac_lr=1e-3,
            actor_lr=1e-3,
            critic_lr=1e-3,
            rollout_steps=512,
            ppo_epochs=5,
            minibatch_size=64,
            log_every_episodes=50,
            rolling_window=50,
        )

        unknown = set(kwargs) - set(train_defaults)
        if unknown:
            raise ValueError(f"Unknown train args: {unknown}")
        train_args = {**train_defaults, **kwargs}

        self._init_optimizers(
            actor_lr=train_args["actor_lr"],
            critic_lr=train_args["critic_lr"],
            shared_ac_lr=train_args["shared_ac_lr"],
        )

        if log_wandb and not WANDB_AVAILABLE:
            warnings.warn("wandb requested but not installed.", RuntimeWarning)
            log_wandb = False

        episode_actor_losses  = []
        episode_critic_losses = []
        episode_entropies     = []
        episode_total_losses  = []

        obs, _        = self.env.reset()
        ep_return     = 0.0
        episode_count = 0
        ep_consumption = 0.0   # sum of all amounts consumed this episode

        while episode_count < train_args["num_episodes"]:

            phy_buf, food_flat_buf                         = [], []
            # x_raw_buf stores unclipped Gaussian samples for PPO log_prob recompute
            act_buf, rew_buf, done_buf, logp_buf, val_buf  = [], [], [], [], []
            x_raw_buf                                       = []

            # ── Rollout ───────────────────────────────────────────────────
            for _ in range(train_args["rollout_steps"]):
                phy, food_flat = obs_to_tensors(obs, self.device)

                amounts, logp, value, mu, log_std, x_raw = self.act(obs, deterministic=False)
                amounts_sq = amounts.squeeze(0)     # (num_foods,)
                logp_sq    = logp.squeeze()
                value_sq   = value.squeeze(-1).squeeze()
                x_raw_sq   = x_raw.squeeze(0)       # (num_foods,)

                # Total consumption this step = sum of all amounts
                ep_consumption += float(amounts_sq.sum().item())

                # Pass numpy array to env
                amounts_np = amounts_sq.detach().cpu().numpy()
                next_obs, reward, terminated, done, info = self.env.step(amounts_np)

                phy_buf.append(phy.squeeze(0))
                food_flat_buf.append(food_flat.squeeze(0))
                act_buf.append(amounts_sq.detach())    # (num_foods,) — clipped amounts
                x_raw_buf.append(x_raw_sq.detach())    # (num_foods,) — unclipped
                rew_buf.append(reward)
                done_buf.append(done)
                logp_buf.append(logp_sq.detach())
                val_buf.append(value_sq.detach())

                ep_return += reward
                obs = next_obs

                if done:
                    self.episode_returns.append(ep_return)
                    self.episode_consumption.append(ep_consumption)
                    episode_count  += 1
                    ep_return       = 0.0
                    ep_consumption  = 0.0
                    obs, _ = self.env.reset()

                    if episode_count % train_args["log_every_episodes"] == 0:
                        rolling_avg   = np.mean(self.episode_returns[-train_args["rolling_window"]:])
                        last_return   = self.episode_returns[-1]
                        last_consump  = self.episode_consumption[-1]
                        avg_consump   = np.mean(self.episode_consumption[-train_args["rolling_window"]:])

                        if printing:
                            print(
                                f"Episode {episode_count:5d} | "
                                f"Last Return: {last_return:8.2f} | "
                                f"Rolling Avg({train_args['rolling_window']}): {rolling_avg:8.2f} | "
                                f"Total Consumption: {last_consump:6.2f} | "
                                f"Avg Consumption: {avg_consump:6.2f} | "
                                f"Distance: {info['distance']:6.4f}"
                            )

                        if log_wandb:
                            wandb.log(
                                {
                                    "train/return":            self.episode_returns[-1],
                                    "train/total_consumption": self.episode_consumption[-1],
                                    "train/actor_loss":        np.mean(episode_actor_losses)  if episode_actor_losses  else 0,
                                    "train/critic_loss":       np.mean(episode_critic_losses) if episode_critic_losses else 0,
                                    "train/entropy":           np.mean(episode_entropies)     if episode_entropies     else 0,
                                    "train/total_loss":        np.mean(episode_total_losses)  if episode_total_losses  else 0,
                                },
                                step=episode_count,
                            )

                    if episode_count >= train_args["num_episodes"]:
                        break

            # ── Bootstrap value ───────────────────────────────────────────
            with torch.no_grad():
                phy_b, food_flat_b = obs_to_tensors(obs, self.device)
                if self.args["shared"]:
                    _, _, val_boot = self.policy(phy_b, food_flat_b)
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
            act_batch       = torch.stack(act_buf)     # (T, num_foods) clipped amounts
            x_raw_batch     = torch.stack(x_raw_buf)  # (T, num_foods) unclipped
            logp_old        = torch.stack(logp_buf)    # (T,)

            # ── PPO update ────────────────────────────────────────────────
            for _ in range(train_args["ppo_epochs"]):
                idx = torch.randperm(len(phy_batch))

                for start in range(0, len(phy_batch), train_args["minibatch_size"]):
                    mb = idx[start : start + train_args["minibatch_size"]]

                    phy_mb  = phy_batch[mb]
                    food_mb = food_flat_batch[mb]

                    if self.args["shared"]:
                        mu_mb, log_std_mb, val = self.policy(phy_mb, food_mb)
                    else:
                        mu_mb, log_std_mb = self.actor(phy_mb, food_mb)
                        val               = self.critic(phy_mb, food_mb)

                    # Recompute log_prob using STORED x_raw (unclipped samples)
                    # This is critical: we must evaluate the policy at the same
                    # unclipped sample, not re-sample, to get a meaningful ratio.
                    logp, entropy = recompute_log_prob(mu_mb, log_std_mb, x_raw_batch[mb])

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

        return self.episode_returns, self.episode_consumption

    # ──────────────────────────────────────────────────────────────────────────
    # Inference / evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_episode(self, log_wandb=False, episode_idx=None):
        """
        Run one full episode with the deterministic (mean) policy.

        Returns (memory_deque, episode_df).

        episode_df columns:
            timestep, reward, value, distance,
            nut_0 … nut_{N-1},          (normalised state)
            nut_0_error … nut_{N-1}_error,
            amount_food_0 … amount_food_{K-1},   (consumption amounts per slot)
            total_consumption                     (sum of amounts at each step)
        """
        env = self.env
        obs, _ = env.reset()
        done   = False

        memory  = deque(maxlen=env.max_steps)
        rewards, values, distances = [], [], []
        phy_states  = []
        # Per-step amounts: list of (num_foods,) arrays
        all_amounts = []

        while not done:
            amounts_t, log_prob, value, mu, log_std, x_raw = self.act(obs, deterministic=True)
            amounts_np = amounts_t.squeeze(0).detach().cpu().numpy()  # (num_foods,)

            next_obs, reward, terminated, done, info = env.step(amounts_np)

            phy = obs["physiological_state"].copy()

            rewards.append(reward)
            values.append(value.item())
            phy_states.append(phy)
            all_amounts.append(amounts_np.copy())
            distances.append(float(np.linalg.norm(phy - env._norm_targets, ord=1)))

            memory.append((
                obs["physiological_state"],
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
        phy_arr    = np.array(phy_states)       # (T, num_nutrients)
        target_arr = env._norm_targets
        names      = env.nutrient_names
        T          = len(rewards)

        col_dict = {
            "timestep": np.arange(T),
            "reward":   np.array(rewards),
            "value":    np.array(values),
            "distance": np.array(distances),
        }

        for i, name in enumerate(names):
            col_dict[name]            = phy_arr[:, i]
            col_dict[f"{name}_error"] = np.abs(phy_arr[:, i] - target_arr[i])

        # Per-slot consumption amounts
        amounts_arr = np.array(all_amounts)     # (T, num_foods)
        for slot_i in range(env.num_foods):
            col_dict[f"amount_slot_{slot_i}"] = amounts_arr[:, slot_i]
        col_dict["total_consumption"] = amounts_arr.sum(axis=1)

        episode_df = pd.DataFrame(col_dict)

        if log_wandb:
            if not WANDB_AVAILABLE:
                warnings.warn("wandb not available – skipping episode logging.", RuntimeWarning)
            else:
                self._log_episode(episode_idx, rewards, values, phy_arr, distances, amounts_arr, env)
                self._log_evaluation_summary(
                    episode_idx=episode_idx,
                    episode_df=episode_df,
                    returns=self.episode_returns,
                    env=env,
                )

        return memory, episode_df

    def infer_episode(self, memory):
        """
        Summarise last state vs target and total consumption per food slot.
        Returns a DataFrame with columns [Nutrient/Slot, Target (normed), Actual (normed) / Total Consumed].
        """
        env   = self.env
        names = env.nutrient_names
        target = env._norm_targets

        last_phy    = memory[-1][0]   # physiological_state (normalised)
        last_amounts = memory[-1][2]  # amounts (num_foods,)

        nutrient_df = pd.DataFrame({
            "Nutrient":         names,
            "Target (normed)":  target,
            "Actual (normed)":  last_phy,
        })

        # Sum total consumption per slot across the full episode
        total_consumed = np.zeros(env.num_foods, dtype=np.float32)
        for entry in memory:
            total_consumed += entry[2]  # amounts at each step

        slot_df = pd.DataFrame({
            "Slot":            [f"slot_{i}" for i in range(env.num_foods)],
            "Total Consumed":  total_consumed,
        })

        return nutrient_df, slot_df

    # ──────────────────────────────────────────────────────────────────────────
    # WandB logging — updated for continuous actions
    # ──────────────────────────────────────────────────────────────────────────

    def _log_episode(self, episode_idx, rewards, values, phy_arr, distances, amounts_arr, env):
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
            # Total consumption per step (sum over all food slots)
            f"inference_{episode_idx}/total_consumption": wandb.plot.line_series(
                xs=steps,
                ys=[amounts_arr.sum(axis=1)],
                keys=["Total consumption"],
                title="Total consumption per step",
                xname="Timestep",
            ),
        }

        # Per-slot consumption lines
        log_dict[f"inference_{episode_idx}/per_slot_consumption"] = wandb.plot.line_series(
            xs=steps,
            ys=[amounts_arr[:, i] for i in range(env.num_foods)],
            keys=[f"slot_{i}" for i in range(env.num_foods)],
            title="Per-slot consumption amounts",
            xname="Timestep",
        )

        wandb.log(log_dict)

        fig = env.plot_consumption(max_time=env.max_steps)
        wandb.log({f"inference_{episode_idx}/consumption_plot": wandb.Image(fig)})
        plt.close(fig)

    def _log_evaluation_summary(self, episode_idx, episode_df, returns, env):

        T = len(episode_df)

        states = np.vstack([
            episode_df[[n for n in env.nutrient_names]].values,
            episode_df[[n for n in env.nutrient_names]].iloc[[-1]].values,
        ])

        rewards   = episode_df["reward"].values
        distances = episode_df["distance"].values

        # Recover amounts array from episode_df
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

        # ── Unnormalise ──────────────────────────────────────────────────────
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

        SKIP_EPS = 15
        WINDOW   = 50

        # Rows: training curve, N nutrient rows, reward/distance, consumption heatmap
        n_rows = n + 3

        fig = plt.figure(figsize=(14, 4.5 * n_rows))
        gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.5, wspace=0.35)

        # ── Training curve ───────────────────────────────────────────────────
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
        ax_tr.annotate(
            f"Max: {max_ret:.2f}", xy=(max_ep, max_ret),
            xytext=(12, -18), textcoords="offset points",
            color="crimson", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
        )
        ax_tr.set_xlabel("Episode")
        ax_tr.set_ylabel("Return")
        ax_tr.set_title(f"PPO Training Curve (first {SKIP_EPS} eps omitted)")
        ax_tr.legend(fontsize=9)
        ax_tr.grid(True, alpha=0.3)

        # ── Nutrient rows ────────────────────────────────────────────────────
        for i, nm in enumerate(names):
            row = i + 1

            # Normalised
            ax_n = fig.add_subplot(gs[row, 0])
            ax_n.axhspan(norm_low[i], norm_high[i], alpha=0.15, color="crimson")
            ax_n.axhline(target_norm[i], ls="--", color="crimson", lw=1.0, alpha=0.5)
            ax_n.plot(t_axis, states[:, i], color="steelblue", lw=1.5, label="Agent state")
            ref_n = np.clip(states[:, i], norm_low[i], norm_high[i])
            ax_n.fill_between(t_axis, states[:, i], ref_n, alpha=0.12, color="steelblue")
            ax_n.set_title(f"{nm} — normalised")
            ax_n.grid(True, alpha=0.3)

            # Raw units
            ax_u = fig.add_subplot(gs[row, 1])
            ax_u.axhspan(unnorm_low[i], unnorm_high[i], alpha=0.15, color="crimson")
            ax_u.axhline(unnorm_targets[i], ls="--", color="crimson", lw=1.0, alpha=0.5)
            ax_u.plot(t_axis, unnorm_states[:, i], color="darkorange", lw=1.5, label="Agent state")
            ref_u = np.clip(unnorm_states[:, i], unnorm_low[i], unnorm_high[i])
            ax_u.fill_between(t_axis, unnorm_states[:, i], ref_u, alpha=0.12, color="darkorange")
            ax_u.set_title(f"{nm} — raw units")
            ax_u.grid(True, alpha=0.3)

        # ── Reward / distance ────────────────────────────────────────────────
        ax_r = fig.add_subplot(gs[n + 1, 0])
        ax_r.plot(t_step, rewards, color="darkorange", lw=1.5)
        ax_r.axhline(0, color="grey", lw=0.8, ls="--")
        ax_r.set_title("Reward per step")
        ax_r.grid(True, alpha=0.3)

        ax_d = fig.add_subplot(gs[n + 1, 1])
        ax_d.plot(t_step, distances, color="teal", lw=1.5)
        ax_d.set_title("Distance to target")
        ax_d.grid(True, alpha=0.3)

        # ── Consumption heatmap (replaces action bar chart) ──────────────────
        # Left: per-slot consumption over time as a heatmap (slots × timesteps)
        ax_ch = fig.add_subplot(gs[n + 2, 0])
        im_c = ax_ch.imshow(
            amounts_arr.T,          # (num_foods, T)
            aspect="auto",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax_ch.set_yticks(range(env.num_foods))
        ax_ch.set_yticklabels([f"slot_{i}" for i in range(env.num_foods)])
        ax_ch.set_xlabel("Timestep")
        ax_ch.set_title("Consumption amounts (slot × time)")
        plt.colorbar(im_c, ax=ax_ch, label="Amount [0, 1]")

        # Right: signed nutrient error heatmap (unchanged)
        below = np.minimum(0.0, states - norm_low[None, :])
        above = np.maximum(0.0, states - norm_high[None, :])
        error = below + above

        ax_h = fig.add_subplot(gs[n + 2, 1])
        im = ax_h.imshow(
            error.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-np.abs(error).max(),
            vmax=np.abs(error).max(),
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
