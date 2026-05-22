
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import warnings
import copy
from collections import deque
from torch.distributions import Categorical
import pandas as pd

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
# Network definitions
# ──────────────────────────────────────────────────────────────────────────────

class SharedActorCritic(nn.Module):
    """
    Shared-trunk actor-critic.

    Input:  physiological_state  (B, num_nutrients)
          + food_embeddings      (B, num_foods * embed_size)  [pre-flattened]
    Output: action_logits (B, num_actions), value (B, 1)
    """

    def __init__(self, state_size, food_flat_size, num_actions, hidden=256, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        in_size = state_size + food_flat_size
        self.trunk  = _mlp([in_size, hidden, hidden])
        self.actor  = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, phy_state, food_emb_flat):
        x = torch.cat([phy_state, food_emb_flat], dim=-1)
        h = self.trunk(x)
        return self.actor(h), self.critic(h)


class Actor(nn.Module):
    def __init__(self, state_size, food_flat_size, num_actions, hidden=256, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = _mlp([state_size + food_flat_size, hidden, hidden, num_actions])

    def forward(self, phy_state, food_emb_flat):
        return self.net(torch.cat([phy_state, food_emb_flat], dim=-1))


class Critic(nn.Module):
    def __init__(self, state_size, food_flat_size, hidden=256, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = _mlp([state_size + food_flat_size, hidden, hidden, 1])

    def forward(self, phy_state, food_emb_flat):
        return self.net(torch.cat([phy_state, food_emb_flat], dim=-1))

