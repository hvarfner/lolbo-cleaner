import numpy as np
from typing import List, Dict, Optional, Callable
import torch 
import sys 
sys.path.append("../")
from lolbo.latent_space_objective import LatentSpaceObjective
from uniref_vae.data import collate_fn
from uniref_vae.load_vae import load_uniref_vae 
from selfies_vae.load_vae import load_selfies_vae
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT
from your_tasks.your_blackbox_constraints import CONSTRAINT_FUNCTIONS_DICT 


class InfoTransformerVAEObjective(LatentSpaceObjective):
    '''Objective class for latent space objectives using InfoTransformerVAE
    '''
    def __init__(
        self,
        task_id: str,
        task_specific_args: List,
        path_to_vae_statedict: str,
        max_string_length: int, # max string length that VAE can generate
        dim: int, # dimension of latent search space 
        constraint_function_ids: List, # list of strings identifying the black box constraint function to use
        constraint_thresholds: List, # list of corresponding threshold values (floats)
        constraint_types: List, # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        vae_load_function: Callable,
        xs_to_scores_dict: Optional[Dict] = {},
        num_calls: Optional[int] = 0,
        init_vae: bool = True, # whether initialize VAE 
    ):
        self.dim                        = dim 
        self.max_string_length          = max_string_length 
        self.path_to_vae_statedict      = path_to_vae_statedict
        self.task_specific_args         = task_specific_args

        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](*self.task_specific_args)

        self.constraint_functions       = []
        for ix, constraint_threshold in enumerate(constraint_thresholds):
            cfunc_class = CONSTRAINT_FUNCTIONS_DICT[constraint_function_ids[ix]]
            cfunc = cfunc_class(
                threshold_value=constraint_threshold,
                threshold_type=constraint_types[ix],
            )
            self.constraint_functions.append(cfunc) 
        self.vae_load_function = vae_load_function
        
        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
            init_vae=init_vae,
        )

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points (bsz, self.dim)
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda() 
        sample = self.vae.sample(z=z.reshape(-1, 2, self.dim//2))
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        return decoded_seqs


    def query_oracle(self, x):
        ''' Input: 
                list of items x (list of aa seqs)
            Output:
                method queries the oracle and returns 
                a LIST of corresponding scores which are y,
                or np.nan in the case that x is an invalid input
                for each item in input list
        '''
        scores_list = self.objective_function(x)
        return scores_list 
    

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.vae, self.dataobj = self.vae_load_function(
            path_to_vae_statedict=self.path_to_vae_statedict,
            dim=self.dim,
            max_string_length=self.max_string_length,
        )


    def vae_forward(self, xs_batch, return_mu_sigma: bool = False, return_dict: bool = False):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae  
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        tokenized_seqs = self.dataobj.tokenize_sequence(xs_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = collate_fn(encoded_seqs)
        dict = self.vae(X.cuda())
        vae_loss, z, z_mu, z_sigma = dict['loss'], dict['z'], dict['mu'], dict['sigma'] 
        z = z.reshape(-1,self.dim) 
        z_mu = z_mu.reshape(-1,self.dim)
        z_sigma = z_sigma.reshape(-1,self.dim)
        if return_dict:
            return dict
        if return_mu_sigma:
            return z, vae_loss, z_mu, z_sigma
        return z, vae_loss

    # black box constraint, treat as oracle 
    @torch.no_grad()
    def compute_constraints(self, xs_batch):
        ''' Input: 
                a list xs (list of sequences)
            Output: 
                c: tensor of size (len(xs),n_constraints) of
                    resultant constraint values, or
                    None of problem is unconstrained
                    Note: constraints, must be of form c(x) <= 0!
        '''
        if len(self.constraint_functions) == 0:
            return None 
        
        all_cvals = []
        for cfunc in self.constraint_functions:
            cvals = cfunc(xs_batch)
            all_cvals.append(cvals)

        return torch.cat(all_cvals, -1)

