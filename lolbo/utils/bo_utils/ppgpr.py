# ppgpr
import math
from .base import DenseNetwork
import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.priors import LogNormalPrior
from botorch.posteriors.gpytorch import GPyTorchPosterior

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(ard_num_dims=256)
        )
        self.covar_module.base_kernel.lengthscale = loc ** 8
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#model.covar_module.base_kernel.lengthscale
    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)
    

class VanillaBOGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 2 ** 0.5, scale: float = 3 ** 0.5):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(VanillaBOGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = (loc + math.log(256) / 2) * 2
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(
                ard_num_dims=256, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
            )
        )
        self.covar_module.base_kernel.lengthscale = scaled_loc
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)

    

class VanillaBONoScaleGPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, loc: float = 2 ** 0.5, scale: float = 3 ** 0.5):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(VanillaBOGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = (loc + math.log(256) / 2) * 2
        self.covar_module = gpytorch.kernels.RBFKernel(
                ard_num_dims=256, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
        )
        self.covar_module.base_kernel.lengthscale = scaled_loc
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)

    

class ExactVanillaBOGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, loc: float = 2 ** 0.5, scale: float = 3 ** 0.5):

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        scaled_loc = loc + math.log(256) / 2
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(
                ard_num_dims=256, 
                lengthscale_prior=LogNormalPrior(loc=scaled_loc, scale=scale)
            )
        )
        self.covar_module.base_kernel.lengthscale = scaled_loc
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)



# gp model with deep kernel
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256) ):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)
