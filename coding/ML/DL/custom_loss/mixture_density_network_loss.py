# The problem and some theory question are in this chat:
# https://claude.ai/share/f5be8b1b-50c0-405f-b681-5b952f9df0d7

import torch
import math
import torch.nn.functional as F

def mdn_loss(
    logits_pi: torch.Tensor, # (B, K)
    means: torch.Tensor, # (B, K, 2)
    log_sigma: torch.Tensor, # (B, K, 2)
    targets: torch.Tensor, # (B, 2)
):
    """
    logits_pi: (batch_size, num_mixtures)
    means: (batch_size, num_mixtures, num_features)
    stds: (batch_size, num_mixtures, num_features)
    targets: (batch_size, num_features)
    """
    D = means.shape[-1]
    log_weights = F.log_softmax(logits_pi, dim=1)
    # above is same as below.
    # log_weights = logits_pi - torch.logsumexp(logits_pi, axis=1, keepdim=True) # B, K
    error = targets[:, None, :] - means # B, K, 2
    per_mixture_loss = - 0.5 * D * math.log(2 * math.pi) - log_sigma.sum(axis=2) - 0.5 * (error * error * torch.exp(-2 * log_sigma)).sum(axis=2) # B, K
    combined_loss = torch.logsumexp(per_mixture_loss + log_weights, dim=-1) # (B, )
    return -combined_loss.mean()

def mdn_loss_trajectory(
    logits_pi: torch.Tensor, # (B, K)
    means: torch.Tensor, # (B, K, T, 2)
    log_sigma: torch.Tensor, # (B, K, T, 2)
    targets: torch.Tensor, # (B, T, 2)
):
    D = means.shape[-1]
    log_weights = F.log_softmax(logits_pi, dim=-1)
    # above is same as below.
    # log_weights = logits_pi - torch.logsumexp(logits_pi, dim=1, keepdim=True) # B, K
    error = targets.unsqueeze(1) - means # B, K, T, 2
    log_sigma = log_sigma.clamp(min=-.7.0, max=7.0)
    log_prob_bkt = - 0.5 * D * math.log(2 * math.pi) - log_sigma.sum(dim=-1) - 0.5 * (error.pow(2) * torch.exp(-2 * log_sigma)).sum(dim=-1) # B, K, T
    log_prob_bk = log_prob_bkt.sum(dim=-1) # (B, K)
    log_prob_b = torch.logsumexp(log_prob_bk + log_weights, dim=-1)
    return -log_prob_b.mean()

    

