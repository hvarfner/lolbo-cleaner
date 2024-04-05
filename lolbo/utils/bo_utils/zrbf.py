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
        if z1_cov is not None and z2_cov is not None:
            A = (self.lengthscale ** 2 + z1_cov + z2_cov).rsqrt() * self.lengthscale
            x1 = z1_mean * A
            x2 = z2_mean * A
            return super().forward(x1, x2) * torch.prod(A, dim=-1)
        
        elif z1_cov is not None:
            assert z1_mean.shape == z1_cov.shape, f"Shape mismatch, mean: {z1_mean.shape}, cov: {z1_cov.shape}"
            #print("insize z1")
            # only applies to diagonal covariances (for now, may generaize if the VAE does to)
            # but for now, we will take the computational saving instead
            B = (self.lengthscale ** 2 + z1_cov).rsqrt() * self.lengthscale
            x1 = z2_mean * B
            return super().forward(x1, z2_mean) * torch.prod(B, dim=-1)

        elif z2_cov is not None:
            assert z2_mean.shape == z2_cov.shape, f"Shape mismatch, mean: {z2_mean.shape}, cov: {z2_cov.shape}"
            # only applies to diagonal covariances (for now, may generaize if the VAE does to)
            # but for now, we will take the computational saving instead
            #print('z2_cov.shape', z2_cov.shape)
            #print('z1_mean.shape', z1_mean.shape)
            B = (self.lengthscale ** 2 + z2_cov).rsqrt() * self.lengthscale
            #print(z2_mean.shape, B.shape)
            x2 = z2_mean * B
            #print('x2.shape', x2.shape)
            #print('both', super().forward(z1_mean, x2).shape, torch.prod(B, dim=-1).shape)
            return super().forward(z1_mean, x2) * torch.prod(B, dim=-1)
    
        else:
            return super().forward(z1_mean, z2_mean)
