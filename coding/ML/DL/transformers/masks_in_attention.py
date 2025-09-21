import torch

# Causal attention mask
# Input the sizes T_q and T_k, sequence lengths of query and key
def make_causal_mask(T_q, T_k):
    return torch.ones(T_q, T_k).tril()

def apply_causal_mask(
        scores: torch.Tensor, # B, T_q, T_k (SHA) or B, N, T_q, T_k (MHA)   (scaled dot product of q and k values)
    ):
        T_q, T_k = scores.shape[-2:]
        mask = make_causal_mask(T_q, T_k)
        mask = mask.expand(*scores.shape)
        return scores.masked_fill_(~mask.bool(), float("-inf"))

T_q, T_k = 3, 5
B, N = 7, 2
scores = torch.rand(B, N, T_q, T_k)
out = apply_causal_mask(scores)
print (out.shape)
print (out)