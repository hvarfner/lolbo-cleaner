import torch
from torch import Tensor
from gpytorch.kernels import RBFKernel

class ZRBFKernel(RBFKernel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, z1, z2, diag=False, **params):
        x1, x2 = self._transform_input_dist(z1, z2)
        # TODO this should be the actual determinant and not the logdet
        return torch.logdet(A).exp() * super().forward(x1, x2, diag, **params)
    
    def _transform_input_dist(z1: Tensor, z2: Tensor):
        pass
        