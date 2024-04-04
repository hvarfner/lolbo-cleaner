from typing import Optional
import torch
from torch import Tensor
from gpytorch.kernels import Kernel, RBFKernel

class ZRBFKernel(RBFKernel):
    
    has_lengthscale = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
            self, 
            z1_mean: Tensor,
            z2_mean: Tensor, 
            z1_cov: Optional[Tensor] = None, 
            z2_cov: Optional[Tensor] = None, 
            diag: bool = False, 
            **params
        ):
        # if cov is not specified, it is assumed that we are passing a latent-space point (i.e.)
        # we revert back to and RBG
        if z1_cov is not None:
            assert z1_cov.ndim == 2
            if z1_cov.ndim == 2:
                z1_cov = torch.diag_embed(z1_cov)

        if z2_cov is not None:
            if z2_cov.ndim == 2:
                z2_cov = torch.diag_embed(z2_cov)
        cov_l = super().forward(z2_mean, z2_mean)
        
        cov


        x1, x2 = self._transform_input_dist(z1, z2)
        # TODO this should be the actual determinant and not the logdet
        return torch.logdet(A).exp() * super().forward(x1, x2, diag, **params)
    
    
    def _transform_input_dist(z1: Tensor, z2: Tensor):
        pass
        