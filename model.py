import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight MLP building block
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(layer_sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# Continuous action helpers
# ──────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -4.0
LOG_STD_MAX =  0.5

def _split_mu_logstd(raw, num_foods):
    """
    Split a (..., num_foods * 2) tensor into mu and log_std, each (..., num_foods).

    mu      : sigmoid(raw[..., :num_foods])        → soft initialisation in (0,1)
    log_std : clamp(raw[..., num_foods:], min, max) → bounded std
    """
    mu_raw  = raw[..., :num_foods]
    ls_raw  = raw[..., num_foods:]
    mu      = torch.sigmoid(mu_raw)                               # (0, 1)
    log_std = torch.clamp(ls_raw, LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std


def sample_continuous_action(mu, log_std):
    """
    Sample consumption amounts from a clipped Gaussian.

    The network predicts mu (mean) and log_std (log standard deviation).
    We sample x_raw ~ N(mu, std), compute log_prob on the UNCLIPPED value
    (standard continuous PPO practice), then hard-clip x to [0, 1] for the env.

    Returns
    -------
    amounts   : (B, num_foods)  float32  in [0, 1]  — clipped samples
    log_prob  : (B,)            float32             — sum log-prob over foods
    entropy   : (B,)            float32             — sum entropy over foods
    x_raw     : (B, num_foods)  float32             — unclipped samples (kept for
                                                      log_prob recomputation during
                                                      PPO update)
    """
    std      = torch.exp(log_std)
    dist     = Normal(mu, std)
    x_raw    = dist.rsample()                          # differentiable sample
    amounts  = torch.clamp(x_raw, 0.0, 1.0)           # clip for environment

    # Log prob computed on UNCLIPPED value — standard approach
    log_prob = dist.log_prob(x_raw).sum(dim=-1)        # scalar per sample
    entropy  = dist.entropy().sum(dim=-1)              # scalar per sample

    return amounts, log_prob, entropy, x_raw


def recompute_log_prob(mu, log_std, x_raw):
    """
    Recompute log_prob and entropy during the PPO update minibatch pass,
    given stored x_raw (unclipped samples from the rollout).

    Parameters
    ----------
    mu, log_std : (B, num_foods)
    x_raw       : (B, num_foods)  — stored unclipped samples from rollout

    Returns
    -------
    log_prob : (B,)
    entropy  : (B,)
    """
    std      = torch.exp(log_std)
    dist     = Normal(mu, std)
    log_prob = dist.log_prob(x_raw).sum(dim=-1)
    entropy  = dist.entropy().sum(dim=-1)
    return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────────────
# Network definitions
# ──────────────────────────────────────────────────────────────────────────────

class SharedActorCritic(nn.Module):
    """
    Shared-trunk actor-critic for CONTINUOUS action space.

    Input:  physiological_state  (B, num_nutrients)
          + food_embeddings      (B, num_foods * embed_size)  [pre-flattened]

    Output:
        raw_ac  : (B, num_foods * 2)  — first num_foods cols = mu_raw,
                                        last  num_foods cols = log_std_raw
        value   : (B, 1)

    The caller is responsible for calling _split_mu_logstd() and
    sample_continuous_action() / recompute_log_prob().
    """

    def __init__(self, state_size, food_flat_size, num_foods, hidden=256, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_foods = num_foods
        in_size        = state_size + food_flat_size
        self.trunk     = _mlp([in_size, hidden, hidden])
        # Outputs mu_raw and log_std_raw concatenated → (num_foods * 2,)
        self.actor_head = nn.Linear(hidden, num_foods * 2)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, phy_state, food_emb_flat):
        x      = torch.cat([phy_state, food_emb_flat], dim=-1)
        h      = self.trunk(x)
        raw_ac = self.actor_head(h)                    # (B, num_foods * 2)
        value  = self.critic_head(h)                   # (B, 1)
        mu, log_std = _split_mu_logstd(raw_ac, self.num_foods)
        return mu, log_std, value


class Actor(nn.Module):
    """
    Separate actor for CONTINUOUS action space.

    Output: mu (B, num_foods), log_std (B, num_foods)
    """

    def __init__(self, state_size, food_flat_size, num_foods, hidden=256, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_foods = num_foods
        in_size        = state_size + food_flat_size
        self.trunk     = _mlp([in_size, hidden, hidden])
        self.head      = nn.Linear(hidden, num_foods * 2)

    def forward(self, phy_state, food_emb_flat):
        h   = self.trunk(torch.cat([phy_state, food_emb_flat], dim=-1))
        raw = self.head(h)                             # (B, num_foods * 2)
        mu, log_std = _split_mu_logstd(raw, self.num_foods)
        return mu, log_std


class Critic(nn.Module):
    """Separate critic — unchanged from discrete version."""

    def __init__(self, state_size, food_flat_size, hidden=256, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = _mlp([state_size + food_flat_size, hidden, hidden, 1])

    def forward(self, phy_state, food_emb_flat):
        return self.net(torch.cat([phy_state, food_emb_flat], dim=-1))
