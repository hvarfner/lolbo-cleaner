from gpytorch.models import GP
from lolbo.utils.bo_utils.ppgpr import (
    GPModel, 
    GPModelDKL, 
    VanillaBOGPModel,
    ExactVanillaBOGPModel
)

def get_model(gp_name: str) -> GP:
    if gp_name == "gp":
        return GPModel
    
    elif gp_name == "dkl":
        return GPModelDKL
    
    elif gp_name == "vanilla":
        return VanillaBOGPModel
    
    elif gp_name == "exact_vanilla":
        return VanillaBOGPModel

    
    else:
        raise ValueError(f"The GP model {gp_name} does not exist.")