from torch import nn
import torch

# Links: 
# Though this may not be a succinct implementation, this is somewhat useful reference for Stanford assignment and the author's code for solving the assignment for different backprops.
#   https://github.com/seloufian/Deep-Learning-Computer-Vision/blob/master/cs231n/assignment2/cs231n/layers.py

# from pytorch, here are the API
# function variant
#   torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
# different module variants
#   torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)  input: N x C or N x C x L (C is num_features) - Temporal Batch Norm
#   torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)  input: N x C x H x W - Spatial Batch Norm
#   torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)  input: N x C x D x H x W - Volumetric Batch Norm
# pytorch reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html

# Assumptions: affine, track_running_stats are assumed default values and not exposed in this implementation. Simple changes to use registrer* methods with None option.
class BatchNorm(nn.Module):
    def __init__(self, num_features, device, dtype, eps=1e-5, momentum=0.1):
        super().__init__()
        tensor_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(self.num_features, **tensor_kwargs))
        self.register_buffer("running_var", torch.ones(self.num_features, **tensor_kwargs))
        self.weight = nn.Parameter(torch.ones(self.num_features, **tensor_kwargs))
        self.bias = nn.Parameter(torch.zeros(self.num_features, **tensor_kwargs))

    def reset_parameters(self):
        # reset the running stats, weight and bias
        # if implemented, we can use torch.empty in the constructor for weight and bias and call this method instead
        pass

    def forward(self, X):
        if X.dim() < 2:
            raise ValueError("expected atleast 2D input")
        if X.size()[1] != self.num_features:
            raise ValueError("mismatched channel dimension")
        expand_shape = (1, ) + (self.num_features,)
        dims = (0, )
        if X.dim() > 2:
            dims = dims + tuple(range(2, X.dim()))
            expand_shape = expand_shape + (1, ) * (X.dim() - 2)
        if self.training:
            mean = torch.mean(X, dim=dims, keepdim=True)
            var = torch.var(X, dim=dims, keepdim=True, correction=0) # this calculates mean again, so inefficient? 
            self.running_mean = self.running_mean * (1-self.momentum) + mean.squeeze() * self.momentum
            self.running_var = self.running_var * (1-self.momentum) + var.squeeze() * self.momentum
        else:
            mean = torch.unflatten(self.running_mean, 0, expand_shape) 
            var = torch.unflatten(self.running_var, 0, expand_shape)
        X = (X - mean) / torch.sqrt(var + self.eps)
        Y = torch.unflatten(self.weight, 0, expand_shape) * X + torch.unflatten(self.bias, 0, expand_shape)
        return Y

if __name__ == "__main__":
    for input_size in [(7, 256), (11, 3, 64, 64), (13, 64, 7, 7, 7)]:
        num_features = input_size[1]
        bn = BatchNorm(num_features, "cpu", torch.float)
        X = torch.randn(*input_size)
        y = bn(X)
        assert y.size() == X.size()
    print ("All tests finished running!")
