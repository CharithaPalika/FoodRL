import torch
import numpy as np
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Bottleneck configuration model
# ──────────────────────────────────────────────────────────────────────────────
# Every network in this file is configured through a small, uniform "spec"
# language instead of a pile of individual *_bottleneck_dim / separate_branches
# kwargs. There are exactly two spec kinds:
#
#   TRUNK spec  (the input-side bottleneck):
#       None                       → no bottleneck (plain 2-layer MLP trunk)
#       int                        → one COMBINED bottleneck over [phy ; food]
#       {"phy": int|None,
#        "food": int|None}         → SEPARATE phy / food branches, each branch
#                                     independently bottlenecked (None on a
#                                     branch = that branch is a plain MLP)
#
#   HEAD spec   (the optional pre-output bottleneck before a head):
#       None                       → head reads the trunk output directly
#       int                        → compress trunk output through this many
#                                     neurons before the head
#
# Each network exposes `self.bottlenecks`: an ordered dict {name → module}
# holding ONLY the bottlenecks that actually exist, so callers can address any
# of them by name (for per-bottleneck amplitude probing) and enumerate what a
# given architecture contains. Local names used here:
#       "trunk"            combined trunk bottleneck
#       "trunk_phy"        phy branch trunk bottleneck (separate mode)
#       "trunk_food"       food branch trunk bottleneck (separate mode)
#       "head"             pre-output bottleneck (standalone Actor / Critic)
#       "actor_head"       actor pre-output bottleneck (shared net)
#       "critic_head"      critic pre-output bottleneck (shared net)
# PPOAgent prefixes these ("actor_trunk", "critic_head", …) into one global
# registry — see agent.py.
# ══════════════════════════════════════════════════════════════════════════════

def normalize_trunk_spec(spec):
    """
    Normalise a trunk spec into (separate_branches, bottleneck_dim,
    phy_bottleneck_dim, food_bottleneck_dim).

    spec : None | int | {"phy": int|None, "food": int|None}
    """
    if spec is None:
        return False, None, None, None
    if isinstance(spec, bool):
        raise ValueError(f"trunk spec must be None, int, or a dict — got bool {spec!r}")
    if isinstance(spec, int):
        return False, spec, None, None
    if isinstance(spec, dict):
        extra = set(spec) - {"phy", "food"}
        if extra:
            raise ValueError(
                f"separate-branch trunk spec accepts only 'phy' and 'food' keys, "
                f"got unexpected {sorted(extra)}."
            )
        return True, None, spec.get("phy"), spec.get("food")
    raise ValueError(
        "trunk spec must be None (no bottleneck), an int (combined bottleneck), "
        f"or a {{'phy':.., 'food':..}} dict (separate branches) — got {type(spec).__name__}."
    )


def normalize_head_spec(spec):
    """Normalise a head spec into a bottleneck dim (int) or None."""
    if spec is None:
        return None
    if isinstance(spec, bool) or not isinstance(spec, int):
        raise ValueError(f"head spec must be None or an int, got {spec!r}.")
    return spec


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight MLP building block
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(layer_sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation
        layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), act()]
    return nn.Sequential(*layers)


def _resolve_per_neuron(value, z_dim, dtype=torch.float32, name="amplitude"):
    """
    Normalize a scalar or 1D vector-like `value` into a (z_dim,) tensor of
    PER-NEURON values. Used for both the multiplicative `amplitude` and the
    additive `bias` applied to a bottleneck activation.

    value : scalar (python int/float, or a 0-d tensor/array) — broadcast to the
                SAME value on every one of the z_dim bottleneck neurons.
            1D sequence/list/ndarray/tensor of length z_dim — one INDEPENDENT
                value per bottleneck neuron.

    Raises ValueError if `value` is neither a scalar nor a length-z_dim vector.
    """
    t = torch.as_tensor(value, dtype=dtype)
    if t.dim() == 0:
        return t.expand(z_dim).clone()
    if t.dim() == 1:
        if t.shape[0] != z_dim:
            raise ValueError(
                f"{name} vector has length {t.shape[0]}, but this bottleneck "
                f"has {z_dim} neurons. Pass a scalar to broadcast the same value "
                f"to every neuron, or a length-{z_dim} vector for per-neuron values."
            )
        return t.clone()
    raise ValueError(
        f"{name} must be a scalar or a 1D vector, got shape {tuple(t.shape)}"
    )


# Backwards-compatible alias (amplitude was the original, only, use).
_resolve_amplitude = _resolve_per_neuron


# ──────────────────────────────────────────────────────────────────────────────
# Bottleneck trunk (single input stream)
#
# bottleneck_dim is not None:
#     in_size --Linear+ReLU--> hidden --Linear+Tanh--> bottleneck_dim (z)
#                                        --Linear+ReLU--> hidden -> (out)
# bottleneck_dim is None (plain MLP, no compression):
#     in_size --Linear+ReLU--> hidden --Linear+ReLU--> hidden (= z)
#
# z is cached on self.z after every forward() so external analysis can read it
# directly (no forward hooks). `amplitude` is a fixed (non-learnable),
# persistent=False PER-NEURON multiplier applied to z right after the encoder —
# a runtime probing knob that is deliberately excluded from state_dict() so it
# is never written into checkpoints. Retune it live via set_amplitude().
# ──────────────────────────────────────────────────────────────────────────────

class _BottleneckTrunk(nn.Module):
    """
    The bottleneck activation is post-processed by two runtime-only knobs:

        z = encoder(x) * amplitude + bias

    `amplitude` (default 1.0) is a multiplicative per-neuron GAIN and `bias`
    (default 0.0) is an additive per-neuron OFFSET. Both are non-learnable,
    `persistent=False` buffers: they move with .to(device) but are deliberately
    excluded from state_dict(), so a probing configuration is never baked into a
    checkpoint. Defaults make both a no-op, so an untouched network behaves
    exactly as if neither existed. Retune live via set_amplitude() / set_bias().
    """

    def __init__(self, in_size, hidden, bottleneck_dim, amplitude=1.0, bias=0.0):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.output_size = hidden
        self.z_dim = bottleneck_dim if bottleneck_dim is not None else hidden
        if bottleneck_dim is not None:
            self.encoder = _mlp([in_size, hidden, bottleneck_dim], output_activation=nn.Tanh)
            self.decoder = _mlp([bottleneck_dim, hidden])
        else:
            self.encoder = _mlp([in_size, hidden, hidden])
            self.decoder = nn.Identity()
        self.register_buffer(
            "amplitude",
            _resolve_per_neuron(amplitude, self.z_dim, name="amplitude"),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            _resolve_per_neuron(bias, self.z_dim, name="bias"),
            persistent=False,
        )
        self.z = None   # populated on each forward(); (B, z_dim)

    def set_amplitude(self, value):
        """Update the per-neuron GAIN in place; takes effect next forward().
        value: scalar (uniform) or length-z_dim vector (per-neuron)."""
        new_amp = _resolve_per_neuron(value, self.z_dim, dtype=self.amplitude.dtype,
                                      name="amplitude")
        self.amplitude.copy_(new_amp.to(self.amplitude.device))

    def set_bias(self, value):
        """Update the per-neuron additive OFFSET in place; takes effect next
        forward(). value: scalar (uniform) or length-z_dim vector (per-neuron)."""
        new_bias = _resolve_per_neuron(value, self.z_dim, dtype=self.bias.dtype,
                                       name="bias")
        self.bias.copy_(new_bias.to(self.bias.device))

    def get_amplitude(self):
        return self.amplitude.detach().cpu().numpy().copy()

    def get_bias(self):
        return self.bias.detach().cpu().numpy().copy()

    def forward(self, x):
        self.z = self.encoder(x) * self.amplitude + self.bias
        return self.decoder(self.z)


# ──────────────────────────────────────────────────────────────────────────────
# Dual-branch trunk (separate phy / food streams)
#
# Each input stream gets its OWN _BottleneckTrunk (own encoder / optional
# bottleneck / decoder). The two decoded outputs are CONCATENATED (not summed)
# so phy-derived and food-derived information stay in separate coordinates all
# the way to the heads. self.z_phy / self.z_food are cached after forward().
# ──────────────────────────────────────────────────────────────────────────────

class _DualBranchTrunk(nn.Module):
    def __init__(self, phy_size, food_size, hidden, phy_bottleneck_dim,
                 food_bottleneck_dim, amplitude=1.0, bias=0.0):
        super().__init__()
        self.phy_branch = _BottleneckTrunk(phy_size, hidden, phy_bottleneck_dim,
                                           amplitude=amplitude, bias=bias)
        self.food_branch = _BottleneckTrunk(food_size, hidden, food_bottleneck_dim,
                                            amplitude=amplitude, bias=bias)
        self.output_size = self.phy_branch.output_size + self.food_branch.output_size
        self.z_phy = None
        self.z_food = None

    def forward(self, phy_state, food_emb_flat):
        decoded_phy = self.phy_branch(phy_state)
        decoded_food = self.food_branch(food_emb_flat)
        self.z_phy = self.phy_branch.z
        self.z_food = self.food_branch.z
        return torch.cat([decoded_phy, decoded_food], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Trunk / head factories that also return a name→module bottleneck registry
# ──────────────────────────────────────────────────────────────────────────────

def _build_trunk(trunk_spec, state_size, food_flat_size, hidden):
    """
    Build the trunk from a trunk spec (see normalize_trunk_spec).

    Returns (trunk_module, output_size, registry, separate_branches) where
    `registry` is an ordered {local_name → _BottleneckTrunk} of the trunk
    bottlenecks that actually exist (a None branch/trunk contributes nothing).
    """
    sep, bdim, pdim, fdim = normalize_trunk_spec(trunk_spec)
    registry = {}
    if sep:
        trunk = _DualBranchTrunk(
            phy_size=state_size, food_size=food_flat_size, hidden=hidden,
            phy_bottleneck_dim=pdim, food_bottleneck_dim=fdim,
        )
        if pdim is not None:
            registry["trunk_phy"] = trunk.phy_branch
        if fdim is not None:
            registry["trunk_food"] = trunk.food_branch
    else:
        trunk = _BottleneckTrunk(state_size + food_flat_size, hidden, bdim)
        if bdim is not None:
            registry["trunk"] = trunk
    return trunk, trunk.output_size, registry, sep


def _build_head_bottleneck(in_size, hidden, head_spec, name):
    """
    Build the optional pre-head bottleneck from a head spec.

    Returns (module_or_None, head_in_size, registry). registry is
    {name → module} if a bottleneck was built, else empty.
    """
    dim = normalize_head_spec(head_spec)
    if dim is None:
        return None, in_size, {}
    module = _BottleneckTrunk(in_size, hidden, dim)
    return module, module.output_size, {name: module}


def _apply_head_bottleneck(module, h):
    """Pass `h` through the pre-head bottleneck if one exists, else identity."""
    return module(h) if module is not None else h


# ──────────────────────────────────────────────────────────────────────────────
# Continuous action helpers
# ──────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -5
LOG_STD_MAX = 5

def _split_mu_logstd(raw, num_foods):
    """Split (..., num_foods*2) into mu=sigmoid(.) and clamped log_std."""
    mu_raw = raw[..., :num_foods]
    ls_raw = raw[..., num_foods:]
    mu = torch.sigmoid(mu_raw)
    log_std = torch.clamp(ls_raw, LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std


def sample_continuous_action(mu, log_std):
    """Sample consumption amounts from a clipped Gaussian. Returns
    (amounts[0,1], log_prob, entropy, x_raw) — log_prob on the UNCLIPPED x_raw."""
    std = torch.exp(log_std)
    dist = Normal(mu, std)
    x_raw = dist.rsample()
    amounts = torch.clamp(x_raw, 0.0, 1.0)
    log_prob = dist.log_prob(x_raw).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    return amounts, log_prob, entropy, x_raw


def recompute_log_prob(mu, log_std, x_raw):
    """Recompute (log_prob, entropy) from stored unclipped x_raw during PPO update."""
    std = torch.exp(log_std)
    dist = Normal(mu, std)
    log_prob = dist.log_prob(x_raw).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────────────
# Discrete action helpers
# ──────────────────────────────────────────────────────────────────────────────

def sample_discrete_action(logits):
    """Sample a discrete action from Categorical(logits). Returns
    (action, log_prob, entropy)."""
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return action, log_prob, entropy


def recompute_discrete_log_prob(logits, action):
    """Recompute (log_prob, entropy) from stored action indices during PPO update."""
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────────────
# Mixin: bottleneck registry access
# ──────────────────────────────────────────────────────────────────────────────

class _BottleneckRegistryMixin:
    """
    Shared helpers for networks that own a `self.bottlenecks` registry
    ({name → _BottleneckTrunk}). Gives every network a uniform way to list,
    read, and retune its bottlenecks by name.
    """

    def bottleneck_names(self):
        return list(self.bottlenecks.keys())

    def get_bottleneck(self, name):
        if name not in self.bottlenecks:
            raise KeyError(
                f"no bottleneck named {name!r} on this network; "
                f"available: {self.bottleneck_names()}"
            )
        return self.bottlenecks[name]

    def set_amplitude(self, name, value):
        """Set the per-neuron gain on the named bottleneck (scalar or vector)."""
        self.get_bottleneck(name).set_amplitude(value)


# ──────────────────────────────────────────────────────────────────────────────
# Network definitions
# ──────────────────────────────────────────────────────────────────────────────

class SharedActorCritic(_BottleneckRegistryMixin, nn.Module):
    """
    Shared-trunk actor-critic for CONTINUOUS actions.

    trunk       : trunk spec (None | int | {"phy":.., "food":..})
    actor_head  : head spec  (None | int) — pre-output bottleneck for the policy head
    critic_head : head spec  (None | int) — pre-output bottleneck for the value head

    Output: raw_ac (B, num_foods*2) split into (mu, log_std); value (B, 1).
    self.bottlenecks names: subset of {trunk|trunk_phy|trunk_food,
    actor_head, critic_head}.
    """

    def __init__(self, state_size, food_flat_size, num_foods, hidden=256,
                 trunk=None, actor_head=None, critic_head=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_foods = num_foods
        self.trunk, trunk_out, reg, self.separate_branches = _build_trunk(
            trunk, state_size, food_flat_size, hidden)
        self.actor_head_bottleneck, actor_in, reg_a = _build_head_bottleneck(
            trunk_out, hidden, actor_head, "actor_head")
        self.critic_head_bottleneck, critic_in, reg_c = _build_head_bottleneck(
            trunk_out, hidden, critic_head, "critic_head")
        self.actor_head = nn.Linear(actor_in, num_foods * 2)
        self.critic_head = nn.Linear(critic_in, 1)
        self.bottlenecks = {**reg, **reg_a, **reg_c}

    def forward(self, phy_state, food_emb_flat):
        if self.separate_branches:
            h = self.trunk(phy_state, food_emb_flat)
        else:
            h = self.trunk(torch.cat([phy_state, food_emb_flat], dim=-1))
        raw_ac = self.actor_head(_apply_head_bottleneck(self.actor_head_bottleneck, h))
        value = self.critic_head(_apply_head_bottleneck(self.critic_head_bottleneck, h))
        mu, log_std = _split_mu_logstd(raw_ac, self.num_foods)
        return mu, log_std, value


class SharedDiscreteActorCritic(_BottleneckRegistryMixin, nn.Module):
    """
    Shared-trunk actor-critic for DISCRETE actions.
    Output: logits (B, num_foods+1); value (B, 1). Spec args as SharedActorCritic.
    """

    def __init__(self, state_size, food_flat_size, num_foods, hidden=256,
                 trunk=None, actor_head=None, critic_head=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_foods = num_foods
        self.trunk, trunk_out, reg, self.separate_branches = _build_trunk(
            trunk, state_size, food_flat_size, hidden)
        self.actor_head_bottleneck, actor_in, reg_a = _build_head_bottleneck(
            trunk_out, hidden, actor_head, "actor_head")
        self.critic_head_bottleneck, critic_in, reg_c = _build_head_bottleneck(
            trunk_out, hidden, critic_head, "critic_head")
        self.actor_head = nn.Linear(actor_in, num_foods + 1)
        self.critic_head = nn.Linear(critic_in, 1)
        self.bottlenecks = {**reg, **reg_a, **reg_c}

    def forward(self, phy_state, food_emb_flat):
        if self.separate_branches:
            h = self.trunk(phy_state, food_emb_flat)
        else:
            h = self.trunk(torch.cat([phy_state, food_emb_flat], dim=-1))
        logits = self.actor_head(_apply_head_bottleneck(self.actor_head_bottleneck, h))
        value = self.critic_head(_apply_head_bottleneck(self.critic_head_bottleneck, h))
        return logits, value


class Actor(_BottleneckRegistryMixin, nn.Module):
    """
    Separate actor for CONTINUOUS actions.
    trunk : trunk spec;  head : head spec.
    self.bottlenecks names: subset of {trunk|trunk_phy|trunk_food, head}.
    Output: mu (B, num_foods), log_std (B, num_foods).
    """

    def __init__(self, state_size, food_flat_size, num_foods, hidden=256,
                 trunk=None, head=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_foods = num_foods
        self.trunk, trunk_out, reg, self.separate_branches = _build_trunk(
            trunk, state_size, food_flat_size, hidden)
        self.head_bottleneck, head_in, reg_h = _build_head_bottleneck(
            trunk_out, hidden, head, "head")
        self.head = nn.Linear(head_in, num_foods * 2)
        self.bottlenecks = {**reg, **reg_h}

    def forward(self, phy_state, food_emb_flat):
        if self.separate_branches:
            h = self.trunk(phy_state, food_emb_flat)
        else:
            h = self.trunk(torch.cat([phy_state, food_emb_flat], dim=-1))
        raw = self.head(_apply_head_bottleneck(self.head_bottleneck, h))
        mu, log_std = _split_mu_logstd(raw, self.num_foods)
        return mu, log_std


class DiscreteActor(_BottleneckRegistryMixin, nn.Module):
    """
    Separate actor for DISCRETE actions.
    trunk : trunk spec;  head : head spec.
    Output: logits (B, num_foods+1). Action 0 = eat nothing; k>=1 = eat slot k-1.
    """

    def __init__(self, state_size, food_flat_size, num_foods, hidden=256,
                 trunk=None, head=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.num_foods = num_foods
        self.trunk, trunk_out, reg, self.separate_branches = _build_trunk(
            trunk, state_size, food_flat_size, hidden)
        self.head_bottleneck, head_in, reg_h = _build_head_bottleneck(
            trunk_out, hidden, head, "head")
        self.head = nn.Linear(head_in, num_foods + 1)
        self.bottlenecks = {**reg, **reg_h}

    def forward(self, phy_state, food_emb_flat):
        if self.separate_branches:
            h = self.trunk(phy_state, food_emb_flat)
        else:
            h = self.trunk(torch.cat([phy_state, food_emb_flat], dim=-1))
        logits = self.head(_apply_head_bottleneck(self.head_bottleneck, h))
        return logits


class Critic(_BottleneckRegistryMixin, nn.Module):
    """
    Separate critic (non-shared mode).
    trunk : trunk spec;  head : head spec.
    self.bottlenecks names: subset of {trunk|trunk_phy|trunk_food, head}.
    """

    def __init__(self, state_size, food_flat_size, hidden=256,
                 trunk=None, head=None, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.trunk, trunk_out, reg, self.separate_branches = _build_trunk(
            trunk, state_size, food_flat_size, hidden)
        self.head_bottleneck, head_in, reg_h = _build_head_bottleneck(
            trunk_out, hidden, head, "head")
        self.value_head = nn.Linear(head_in, 1)
        self.bottlenecks = {**reg, **reg_h}

    def forward(self, phy_state, food_emb_flat):
        if self.separate_branches:
            h = self.trunk(phy_state, food_emb_flat)
        else:
            h = self.trunk(torch.cat([phy_state, food_emb_flat], dim=-1))
        return self.value_head(_apply_head_bottleneck(self.head_bottleneck, h))
