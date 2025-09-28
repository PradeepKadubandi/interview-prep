import torch

# Causal attention mask
# Input the sizes T_q and T_k, sequence lengths of query and key
def make_causal_mask(T_q, T_k):
    return torch.ones(T_q, T_k).tril()

def make_block_causal_mask(T_q, T_k_past):
      # In speculative decoding with KV cache,
      # we have past_key_values of sequence len T_k_past - the queries should attend to all these
      # T_q is the number of tokens in spec decode - so len of query tokens
      # we should have causal mask for these T_q tokens
      # Total tensor should be of shape (T_q, T_k_past + T_q)
      cache_mask = torch.ones(T_q, T_k_past)
      query_mask = make_causal_mask(T_q, T_q)
      total_mask = torch.cat((cache_mask, query_mask), dim=1)
      return total_mask

def apply_causal_mask(
        scores: torch.Tensor, # B, T_q, T_k (SHA) or B, N, T_q, T_k (MHA)   (scaled dot product of q and k values)
        test_block_causal: bool # whether mask should be for block attention or causal attention
    ):
        T_q, T_k = scores.shape[-2:]
        if test_block_causal:
            mask = make_block_causal_mask(T_q, T_k - T_q) # Speculative decoding, T_k here is total k size, so past_k size is T_k - T_q
        else:
            mask = make_causal_mask(T_q, T_k) # regular context encoding or training
        mask = mask.expand(*scores.shape)
        return scores.masked_fill_(~mask.bool(), float("-inf"))

T_q, T_k = 3, 5
B, N = 7, 2
# Set the below bool to fale to test the regular causal attention mask
test_block_causal = True
if test_block_causal:
    scores = torch.rand(B, N, T_q, T_k + T_q) # shape of attention scores in speculative decoding
else:
    scores = torch.rand(B, N, T_q, T_k) # attention scores are this shape in training or regular context encoding
out = apply_causal_mask(scores, test_block_causal)
print (out.shape)
print (out)