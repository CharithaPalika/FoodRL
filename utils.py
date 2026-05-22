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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def choose_action(logits):
    dist    = Categorical(logits=logits)
    action  = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def obs_to_tensors(obs, device):
    """
    Convert a FoodEnv observation dict to tensors.
    Returns:
        phy       : (1, num_nutrients)
        food_flat : (1, num_foods * embed_size)   – flattened food embeddings
    """
    phy = torch.tensor(
        obs["physiological_state"], dtype=torch.float32, device=device
    ).unsqueeze(0)                                         # (1, N)

    food_flat = torch.tensor(
        obs["food_embeddings"], dtype=torch.float32, device=device
    ).flatten().unsqueeze(0)                               # (1, num_foods*embed_size)

    return phy, food_flat

