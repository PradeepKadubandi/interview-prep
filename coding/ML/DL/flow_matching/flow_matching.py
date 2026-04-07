# Train: sample noise from unit normal
#   sample time from beta but remove the two extremes
#   ground truth actions -> input, t = 0
#   interpolated state x_t = t * noise + (1-t) * actions
#   gt velocity = actions - noise (assuming we go from t=1 to t=0)
#   predicted velocity = net(interpolated state, sampled time)

from torch.distributions import Beta, Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

def sample_time(batch_size):
    time = Beta(concentration1=1.5, concentration0=1.0).sample((batch_size,))
    print (f"In sample_time: {time.shape=}")
    print (f"In sample_time: {time=}")
    time = time * 0.999 + .001
    return time

def sample_noise(gt_actions):
    return torch.normal(mean=0.0, std=1.0, size=gt_actions.shape)

def predict_step(gt_actions):
    noise = sample_noise(gt_actions)
    print (f"{noise.shape=}")
    time = sample_time(gt_actions.shape[0])
    time_expanded = time[:, None, None]  # expand to (B, 1, 1)
    print (f"In predicte_step after expand: {time_expanded.shape=}")
    x_t = noise * time_expanded + (1 - time_expanded) * gt_actions
    gt_v = noise - gt_actions
    predicted_v = net(x_t, time) # <-- time is sent as (B,) here and used to create a sinusoidal embedding inside net
    return predicted_v, gt_v


# Copied from modeling_pi0.py in lerobot
def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

class SimpleNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.projection_dim = 128
        self.action_in_proj = nn.Linear(action_dim, self.projection_dim)
        self.action_time_mlp_in = nn.Linear(2 * self.projection_dim, self.projection_dim)
        self.action_time_mlp_out = nn.Linear(self.projection_dim, action_dim) # in pi0, this actually projects to projection_dim because that's the token dim in transformer

    def forward(self, x, t):
        # x: (B, horizon_length, action_dim)
        # t: (B,)
        B, H, D = x.shape
        t = create_sinusoidal_pos_embedding(t, dimension=self.projection_dim, min_period=4e-3, max_period=4.0, device=x.device)
        print (f"In forward after create_sinusoidal_pos_embedding: {t.shape=}")
        action_emb = self.action_in_proj(x)  # (B, horizon_length, projection_dim)
        print (f"In forward after fc1: {action_emb.shape=}")
        t = t[:, None, :].expand_as(action_emb)  # expand to (B, horizon_length, projection_dim)
        print (f"In forward after expand: {t.shape=}")
        x = torch.cat([action_emb, t], dim=2)  # concatenate time to each action, can be dim=-1
        print (f"In forward after concat: {x.shape=}")
        x = F.silu(self.action_time_mlp_in(x))
        x = self.action_time_mlp_out(x)
        return x

batch_size = 3
horizon_length = 4
action_dim = 16
gt_actions = torch.rand((batch_size, horizon_length, action_dim))
print (f"{gt_actions.shape=}")
net = SimpleNet(action_dim)
predicted_v, gt_v = predict_step(gt_actions)
print (f"{predicted_v.shape=}")
loss = F.mse_loss(predicted_v, gt_v)
