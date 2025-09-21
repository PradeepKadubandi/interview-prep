# Use torch to implement multi-head attention

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    # Other possible concerns can be seen from this interface: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    # dropout, bias support etc
    # simplifying the code to not handle device or the above options...
    def __init__(self,
                 hidden_size,
                 n_heads):
        super().__init__()
        self.hidden_size = hidden_size # This can be also named d_model
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, "heads must evenly divide hidden size"
        self.head_dim = self.hidden_size // self.n_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # initialize to _xavier_uniform

    def reshape_to_heads(self, x):
        # B,S,H -> B, n, S, d
        B, S, H = x.shape
        assert H == self.hidden_size
        return x.view(B, S, self.n_heads, self.head_dim).permute((0, 2, 1, 3)) # contiguous?
    
    def reshape_from_heads(self, x):
        # B, n, S, d to B,S,H
        B, n, S, d = x.shape
        assert n == self.n_heads
        assert d == self.head_dim
        return x.permute((0, 2, 1, 3)).contiguous().view(B, S, self.hidden_size)
    

    def forward(self,
            x_q: torch.Tensor, # B, S_q, H
            x_kv: Optional[torch.Tensor] = None # B, S_k, H
        ):
        if x_kv is None:
            x_kv = x_q # self attention

        # B, n, S, d -> B, n, S, d
        q_values = self.q_proj(x_q)
        k_values = self.k_proj(x_kv)
        v_values = self.v_proj(x_kv)

        q_values = self.reshape_to_heads(q_values)
        k_values = self.reshape_to_heads(k_values)
        v_values = self.reshape_to_heads(v_values)

        scores = torch.einsum("bnqd,bnsd->bnqs", q_values, k_values)
        # return scores
        scores = F.softmax(scores / math.sqrt(self.head_dim), dim=-1)

        scores = torch.einsum("bnqs,bnsd->bnqd", scores, v_values)
        scores = self.reshape_from_heads(scores)
        # return scores
        
        o_values = self.o_proj(scores)
        return o_values
        


B, S, H = 7, 8, 256
n_heads = 2
test_ip = torch.Tensor(B, S, H)
test_class = MultiHeadAttention(H, n_heads)
output = test_class(test_ip)
print (output.shape)