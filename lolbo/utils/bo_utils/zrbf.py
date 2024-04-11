from typing import Optional, Union
import torch
from torch import Tensor
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
    

class ZRBFKernel(RBFKernel):
    
    has_lengthscale = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_z(self, input, z_check):
        return torch.any(torch.all(input == z_check.unsqueeze(-2), dim=-1), dim=0)

    def forward(
            self, 
            z1_mean: Tensor, 
            z2_mean: Tensor, 
            z_cov: Optional[Tensor] = None,
            is_z: Optional[Tensor] = None,
            diag: bool = False, 
            **params
        ):
        # if cov is not specified, it is assumed that we are passing a latent-space point (i.e.)
        # we revert back to and RBG
        if is_z is None:
            check_z1 = Tensor([False])
            check_z2 = Tensor([False])
        else:
            check_z1 = self._compute_z(z1_mean, is_z)
            check_z2 = self._compute_z(z2_mean, is_z) 
        
        if torch.any(check_z1) and not torch.all(check_z1):
            raise ValueError("Not all elements in array 1 are strictly x or z!")
        if torch.any(check_z2) and not torch.all(check_z2):
            raise ValueError("Not all elements in array 2 are strictly x or z!")
        if torch.all(check_z1) and torch.all(check_z2):
            A = (self.lengthscale ** 2 + 2 * z_cov).rsqrt() * self.lengthscale
            x1 = z1_mean * A
            x2 = z2_mean * A

            return super().forward(x1, x2, diag=diag) * torch.prod(A, dim=-1)

        elif torch.all(check_z1):
            assert z1_mean.shape == z_cov.shape, f"Shape mismatch, mean: {z1_mean.shape}, cov: {z_cov.shape}"
            # only applies to diagonal covariances (for now, may generaize if the VAE does to)
            # but for now, we will take the computational saving instead
            B = (self.lengthscale ** 2 + z_cov).rsqrt() * self.lengthscale
            x1 = z2_mean * B
            return super().forward(x1, z2_mean, diag=diag) * torch.prod(B, dim=-1)

        elif torch.all(check_z2):
            assert z2_mean.shape == z_cov.shape, f"Shape mismatch, mean: {z2_mean.shape}, cov: {z_cov.shape}"
            # only applies to diagonal covariances (for now, may generaize if the VAE does to)
            # but for now, we will take the computational saving instead
            B = (self.lengthscale ** 2 + z_cov).rsqrt() * self.lengthscale
            x2 = z2_mean * B
            return super().forward(z1_mean, x2, diag=diag) * torch.prod(B, dim=-1)
    
        else:
            return super().forward(z1_mean, z2_mean, diag=diag)
