import math
import os
import sys
import random
import matplotlib.pyplot as plt
sys.path.append("../")
import torch
from torch.distributions import Normal
import numpy as np 
import pandas as pd
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
from lolbo.utils.pred_utils import (
    get_train_test_split, 
    compute_mll, 
    compute_rmse
)
from lolbo.utils.bo_utils.ppgpr import ZGPModel
from lolbo.utils.bo_utils.zrbf import ZRBFKernel
from gpytorch.models import GP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from lolbo.lolbo import LOLBOState
from lolbo.utils.bo_utils.registry import get_model
from lolbo.latent_space_objective import LatentSpaceObjective
import signal 
import copy 
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False


class Optimize(object):
    """
    Run LOLBO Optimization
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run LOLBO). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
    """
    def __init__(
        self,
        task_id: str,
        seed: int=42,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        num_iter: int=100_000,
        vae_learning_rate: float=0.001,
        gp_learning_rate: float=0.001,
        acq_func: str="ts",
        bsz: int=16,
        num_init: int=10_000,
        init_n_update_epochs: int=20,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        update_e2e: bool=True,
        k: int=1_000,
        verbose: bool=True,
        recenter_only=False,
        log_table_freq=10_000, 
        save_vae_ckpt=False,
        gp_name: GP = "dkl",
        freeze_vae: bool = False,
        query_at_recenter: bool = True,
        train_from_pretrained: bool = False,
        experiment_name: str = "test",

    ):
        signal.signal(signal.SIGINT, self.handler)
        # add all local args to method args dict to be logged by wandb
        self.save_vae_ckpt = save_vae_ckpt
        self.log_table_freq = log_table_freq # log all collcted data every log_table_freq oracle calls 
        self.recenter_only = recenter_only # only recenter, no E2E 
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = num_iter
        self.verbose = verbose
        self.num_initialization_points = num_init
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e
        self.model = gp_name
        self.experiment_name = experiment_name
        self.set_seed()
        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = f"optimimze-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"

        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and self.init_train_z
        self.load_train_data(seed=self.seed)
        # initialize latent space objective (self.objective) for particular task
        self.initialize_objective()
        assert isinstance(self.objective, LatentSpaceObjective), "self.objective must be an instance of LatentSpaceObjective"
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        if self.init_train_c is not None: # if constrained 
            assert torch.is_tensor(self.init_train_c), "load_train_data() must set self.init_train_c to a tensor of cs"
            assert self.init_train_c.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of cs and xs, instead got {len(self.init_train_x)} xs and {self.init_train_c.shape[0]} cs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert self.init_train_y.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of ys and xs, instead got {self.init_train_y.shape[0]} ys and {len(self.init_train_x)} xs"
        assert self.init_train_z.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of zs and xs, instead got {self.init_train_z.shape[0]} zs and {len(self.init_train_x)} xs"
        
        gp = get_model(gp_name)
        # initialize lolbo state
        self.lolbo_state = LOLBOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            train_c=self.init_train_c,
            gp=gp,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            vae_learning_rate=vae_learning_rate,
            gp_learning_rate=gp_learning_rate,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose,
            freeze_vae=freeze_vae,
            query_at_recenter=query_at_recenter,
            train_from_pretrained=train_from_pretrained,

        )


    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self


    def load_train_data(self, seed: int = 42):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_c (a tensor of constraint values/c's)
                self.init_train_z (a tensor of corresponding latent space points)
        '''
        return self


    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self


    def create_wandb_tracker(self):
        if self.track_with_wandb:
            config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config=config_dict,
            ) 
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self


    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb: 
            dict_log = {
                "best_found":self.lolbo_state.best_score_seen,
                "n_oracle_calls":self.lolbo_state.objective.num_calls,
                "total_number_of_e2e_updates":self.lolbo_state.tot_num_e2e_updates,
                "best_input_seen":self.lolbo_state.best_x_seen,
            }
            dict_log[f"TR_length"] = self.lolbo_state.tr_state.length
            self.tracker.log(dict_log) 

        return self

    def run_lolbo(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        last_logged_n_calls = 0 # log table + save vae ckpt every log_table_freq oracle calls
        #main optimization loop
        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                if not self.recenter_only:
                    self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
                if self.recenter_only:
                    self.lolbo_state.update_surrogate_model()
            else: # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            if (self.lolbo_state.objective.num_calls - last_logged_n_calls) >= self.log_table_freq:
                self.final_save = False 
                self.log_topk_table_wandb()
                last_logged_n_calls = self.lolbo_state.objective.num_calls


        # if verbose, print final results 
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # log top k scores and xs in table
        self.final_save = True 
        self.log_topk_table_wandb()
        self.tracker.finish()

        return self 


    def run_vanilla(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        #main optimization loop
        self.lolbo_state.initial_surrogate_model_update()
        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
                self.save_to_csv()
            
            #else: # otherwise, just update the surrogate model on data
            #    self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            self.save_to_csv()
            print(self.lolbo_state.objective.num_calls, len(self.lolbo_state.orig_train_y), self.max_n_oracle_calls)
        self.save_to_csv()
        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        return self  

    def save_to_csv(self, save_best_nbr: int = 1000):
        save_dir = os.environ.get("SAVE_DIR", "..")
        res_save_path = f"{save_dir}/result_values/{self.experiment_name}/{self.task_id}/{self.model}"
        str_save_path = f"{save_dir}/result_strings/{self.experiment_name}/{self.task_id}/{self.model}"
        os.makedirs(res_save_path, exist_ok=True)
        os.makedirs(str_save_path, exist_ok=True)
        Y = self.lolbo_state.orig_train_y.flatten()
        df = pd.DataFrame({self.task_id: Y})
        df.to_csv(f"{res_save_path}/{self.task_id}_{self.model}_{self.seed}.csv")
        best_indices = torch.argsort(Y, descending=True)[:save_best_nbr]
        best_X = np.array(self.lolbo_state.train_x)[best_indices].tolist()
        best_y = Y[best_indices]
        df_indices = pd.DataFrame({"string": best_X, "{self.task_id}": best_y})
        df.to_csv(f"{str_save_path}/{self.task_id}_{self.model}_{self.seed}_strings.csv")

    def run_prediction(
            self, 
            use_train: bool = True
        ):
        last_logged_n_calls = 0 # log table + save vae ckpt every log_table_freq oracle calls
        #main optimization loop 
        self.load_test_data(self.seed)
        self.log_data_to_wandb_on_each_loop()
        # update models end to end when we fail to make
        print("Updating model end to end")
        self.lolbo_state.update_models_e2e()
        print("Updated")
        
        test_x = self.init_train_x
        test_y = self.init_train_y.flatten()
        data = {
            "train": (self.init_train_x, self.init_train_y.flatten()),
            "test": (self.test_x, self.test_y.flatten())
        }
        for key, (test_x, test_y) in data.items():
            gp = self.lolbo_state.model 
            print(gp.covar_module.base_kernel.lengthscale)

            bsize = self.lolbo_state.bsz
            num_test = len(test_x)
            num_batches = math.ceil(num_test / bsize)
            # squared error
            gp.eval()
            # marginal loglik
            mean, std = self.lolbo_state.ymean, self.lolbo_state.ystd
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            y_decoded = torch.zeros(len(test_y))
            pred_mll = -10 * torch.ones(len(test_x))
            decoded_pred_mll = -10 * torch.ones(len(test_x))
            decoded_pred_se = -10 * torch.ones(len(test_x))
            pred_se = -10 * torch.ones(len(test_x))
            pred_means = torch.ones(len(test_x))
            pred_stds = torch.ones(len(test_x))
            valid = torch.ones(len(test_x)).to(torch.long)
            exact_match_count = torch.zeros(len(test_x)).to(torch.bool)
            for batch_idx in range(num_batches):
                print(f"{batch_idx+1}/{num_batches}")
                batch_lb, batch_ub = batch_idx * bsize, min(num_test, (batch_idx + 1) * bsize)
                x_batch = test_x[batch_lb: batch_ub]
                _, vae_loss, z_mu, z_sigma = self.objective.vae_forward(x_batch, return_mu_sigma=True)
                y_batch = test_y[batch_lb:batch_ub].to(z_mu)
                output_z = self.objective(z_mu)
                decoded_y_batch = torch.Tensor(output_z["scores"]).to(z_mu).flatten()
                decoded_y_batch = torch.nan_to_num(decoded_y_batch, -0.1)
                valid[batch_lb: batch_ub] = torch.Tensor(output_z["bool_arr"])
                
                for i, batch_nbr in enumerate(range(batch_lb, batch_ub)):
                    if output_z["decoded_xs"][i] == x_batch[i]:
                        exact_match_count[batch_nbr] = 1
                        print(exact_match_count.sum())
                y_decoded[batch_lb: batch_ub] = decoded_y_batch    
                post = gp.posterior(z_mu)
                
                pred_means[batch_lb: batch_ub] = post.mean.cpu().detach().squeeze(-1)
                pred_stds[batch_lb: batch_ub] = post.variance.sqrt().cpu().detach().squeeze(-1)
                
                norm = Normal(post.mean.squeeze(-1), post.variance.sqrt().squeeze(-1))
                pred_se[batch_lb: batch_ub] = std * torch.pow(
                    post.mean.squeeze(-1) - (y_batch - mean) / std, 2
                ).cpu().detach()
                decoded_pred_se[batch_lb: batch_ub] = std * torch.pow(
                    post.mean.squeeze(-1) - (decoded_y_batch - mean) / std, 2
                ).cpu().detach()
                norm_obs = (y_batch - mean) / std
                decoded_norm_obs =  (decoded_y_batch - mean) / std

                pred_mll[batch_lb: batch_ub] = norm.log_prob((norm_obs)).cpu().detach().flatten()
                decoded_pred_mll[batch_lb: batch_ub] = norm.log_prob((decoded_norm_obs)).cpu().detach().flatten()
            pred_mll = pred_mll - torch.log(std)

            pred_mll = pred_mll[valid]
            decoded_pred_mll = decoded_pred_mll[valid]
            pred_se = pred_se[valid]
            decoded_pred_se = decoded_pred_se[valid]

            mll_mean, decoded_mll_mean = pred_mll.mean().item(), decoded_pred_mll.mean().item() 
            rmse, decoded_rmse = pred_se.mean().sqrt().item(), decoded_pred_se.mean().sqrt().item()
            
            decoding_rmse = torch.pow(y_decoded - test_y, 2).mean().sqrt()
            lengthscale = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().tolist()
            res = {"mll_mean": [mll_mean], "decoded_mll_mean": [decoded_mll_mean], "rmse": rmse, "decoded_rmse": decoded_rmse}
            recon_std = torch.abs(test_y - y_decoded)[valid].std()
            #for i, ls in enumerate(lengthscale):
            #    res[f"ls_{i}"] = ls
            
            sort_order = torch.sort(test_y).indices
            sorted_output = test_y[sort_order]
            sorted_means = pred_means[sort_order] * std + mean
            sorted_stds = pred_stds[sort_order] * std
            diff = sorted_output[-1] - sorted_output[0]
            range_ = sorted_output[0] - 0.1 * diff, sorted_output[-1] + 0.1 * diff 
            print(valid.sum(), (~valid).sum())
            plt.rcParams['font.family'] = 'serif'
            os.makedirs(f"../pred_results/{self.task_id}/{self.gp_name}", exist_ok=True)
            with torch.no_grad():
                plt.plot(range_, range_, color='blue', linestyle="dashed")
                plt.vlines(sorted_output, sorted_means - 2 * sorted_stds, sorted_means + 2 * sorted_stds, color="grey", alpha=0.15, linewidth=0.1)
                plt.scatter(sorted_output, sorted_means, color="black", s=5)
                plt.xlabel("Actual values")
                plt.ylabel("Predicted values")
                plt.title(f"{self.gp_name}_{self.task_id}_{self.seed}\nRMSE: {round(rmse * std.item(), 5)}, -- MLL: {round(mll_mean, 3)}", fontsize=16)    
                plt.grid(True)
                plt.savefig(f"../pred_results/{self.task_id}/{self.gp_name}/pred_{self.gp_name}_{self.task_id}_{self.seed}_{key}.pdf")
                plt.clf()
                plt.plot(range_, range_, color='blue', linestyle="dashed")
                plt.scatter(sorted_output, y_decoded[sort_order], color="red", s=5)
                plt.xlabel("Actual values")
                plt.ylabel("Decoded values")
                plt.title(f"{self.gp_name}_{self.task_id}_{self.seed}\nRMSE: {recon_std.item()}", fontsize=16)    
                plt.grid(True)
                plt.savefig(f"../pred_results/{self.task_id}/{self.gp_name}/{self.gp_name}_{self.task_id}_{self.seed}_{key}_decoded_err.pdf")
                plt.clf()
            pd.DataFrame(res).to_csv(f"../pred_results/{self.task_id}/{self.gp_name}/pred_{self.gp_name}_{self.task_id}_{self.seed}_{key}.csv")
    
        # Testing sampled value
        encoded_test_z = torch.randn((num_test, z_mu.shape[1])).to(z_mu)
        for batch_idx in range(num_batches):
            print(f"{batch_idx+1}/{num_batches}")
            batch_lb, batch_ub = batch_idx * bsize, min(num_test, (batch_idx + 1) * bsize)
            output_z = self.objective(encoded_test_z[batch_lb:batch_ub])
            decoded_y_batch = torch.Tensor(output_z["scores"]).to(z_mu).flatten()
            decoded_y_batch = torch.nan_to_num(decoded_y_batch, -0.1)
            valid[batch_lb: batch_ub] = torch.Tensor(output_z["bool_arr"])
            
            y_decoded[batch_lb: batch_ub] = decoded_y_batch    
            post = gp.posterior(encoded_test_z[batch_lb:batch_ub])
            
            pred_means[batch_lb: batch_ub] = post.mean.cpu().detach().squeeze(-1)
            pred_stds[batch_lb: batch_ub] = post.variance.sqrt().cpu().detach().squeeze(-1)
        
        sort_order = torch.sort(y_decoded).indices  
        sorted_output = y_decoded[sort_order]
        sorted_means = pred_means[sort_order] * std + mean
        sorted_stds = pred_stds[sort_order] * std
        plt.plot(range_, range_, color='blue', linestyle="dashed")
       # plt.vlines(sorted_output, sorted_means - 2 * sorted_stds, sorted_means + 2 * sorted_stds, color="grey", alpha=0.15, linewidth=0.1)
        plt.scatter(sorted_output, sorted_means, color="green", s=5)
        plt.ylabel("Predicted values")
        plt.xlabel("Decoded values")
        plt.title(f"{self.gp_name}_{self.task_id}_{self.seed}\nRMSE: {recon_std.item()}", fontsize=16)    
        plt.grid(True)
        plt.savefig(f"../pred_results/{self.task_id}/{self.gp_name}/{self.gp_name}_{self.task_id}_{self.seed}_{key}_pred_err.pdf")


    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end 
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.lolbo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.lolbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.lolbo_state.objective.num_calls}")

        return self
    

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we log top xs, scores found
        print("Ctrl-c hass been pressed, wait while we save all collected data...")
        self.final_save = True 
        self.log_topk_table_wandb()
        print("Now terminating wandb tracker...")
        self.tracker.finish() 
        msg = "Data now saved and tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)


    def log_topk_table_wandb(self):
        ''' After optimization finishes, log
            top k inputs and scores found
            during optimization '''
        if self.track_with_wandb and self.final_save:
            # save top k xs and ys 
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.top_k_scores):
                data_list.append([ score, str(self.lolbo_state.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})

            # Additionally save table of ALL collected data 
            cols = ['All Scores', "All Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.train_y.squeeze()):
                data_list.append([ score.item(), str(self.lolbo_state.train_x[ix]) ])
            try:
                full_table = wandb.Table(columns=cols, data=data_list)
                self.tracker.log({f"full_table": full_table})
            except:
                self.tracker.log({'save-data-table-failed':True})
            
        # We also want to save the fine-tuned VAE! 
        if self.save_vae_ckpt:
            n_calls = self.lolbo_state.objective.num_calls
            model = copy.deepcopy(self.lolbo_state.objective.vae)
            model = model.eval()
            model = model.cpu() 
            save_dir = 'finetuned_vae_ckpts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir, exist_ok=True)
            model_save_path = save_dir + self.wandb_project_name + '_' + wandb.run.name + f'_finedtuned_vae_state_after_{n_calls}evals.pkl'  
            torch.save(model.state_dict(), model_save_path) 

        save_dir = 'optimization_all_collected_data/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_path = save_dir + self.wandb_project_name + '_' + wandb.run.name + '_all-data-collected.csv'
        df = {}
        df['train_x'] = np.array(self.lolbo_state.train_x)
        df['train_y'] = self.lolbo_state.train_y.squeeze().detach().cpu().numpy()  
        df = pd.DataFrame.from_dict(df)
        df.to_csv(file_path, index=None) 

        return self


    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)
