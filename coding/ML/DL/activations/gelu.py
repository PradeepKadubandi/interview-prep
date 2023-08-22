# History: Asked Chat GPT for implementation and improved it according to my needs with some follow up prompts.
#   Below is a good implemenation.
# Also, this can be the reference for how an operator (along with forward pass and backward pass) can be structured (static methods in a class) in a pytorch based implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = 0.5 * input * (1.0 + torch.erf(input / math.sqrt(2.0)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        cdf = 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * torch.pow(input, 2)) / math.sqrt(2 * math.pi)
        derivative = 0.5 + 0.5 * torch.erf(input / math.sqrt(2.0)) + input * pdf
        return grad_output * cdf * derivative

class GELU(nn.Module):
    def forward(self, input):
        return GELUFunction.apply(input)

# Example usage
input = torch.randn(10, requires_grad=True)  # Example input
gelu = GELU()
output = gelu(input)
loss = output.sum()  # Example loss
loss.backward()
print(input.grad)
