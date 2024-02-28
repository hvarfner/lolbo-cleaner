# Applying Deep Bayesian Optimizatoin Using LOL-BO and ROBOT 

This repository is for easy application of the Deep BO algorithms LOL-BO (Local Latent-space Bayesian Optimization) and ROBOT (Rank Ordered Bayesian Optimization with Trust Regions) algorithms for optimization of any black-box objectives over structured input spaces. 
The repository can use any VAE to map the structured input space to a continuous latent search space and any black-box objective desired. 
Directions below explain how to set up your VAE and objective function to run LOL-BO and ROBOT.  

## LOL-BO (Local Latent-space Bayesian Optimization)
LOL-BO is a Deep Bayesian Optimization algrotihm that can be used to search over the latent space of a VAE to find an optima which maximzes some black-box objective defired over some structured input space (i.e. the space of molecular strings or protein sequences). 
See LOL-BO paper here: https://arxiv.org/abs/2201.11872

## ROBOT (Rank Ordered Bayesian Optimization with Trust Regions)
ROBOT is a Deep Bayesian Optimization algrotihm that can be used to search over the latent space of a VAE to find a set of diverse optima, all of which maximzes the given black-box objective function. For ROBOT, we must also define some diverstiy function divf which measures the diversity between two objects in the structured search space. ROBOT finds a diverse set of M optima such that all pairs of optima in the set are diverse according to the specified diversity function divf and specified diversity threshold tau. 
See ROBOT paper here: https://arxiv.org/abs/2210.10953 

## Weights and Biases (wandb) tracking
This repo it set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
Otherwise, the code can also be run without wandb tracking by simply setting the argument `--track_with_wandb False` (see example commands below). 

## Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

## Setting up your optimization task  
Modify the following files to set up your desired optimization problem - your desired VAE, objective function(s), black-box constrains if any, and diversity function for running ROBOT. 

1. your_tasks/your_objective_function.py

Defines a dictionary OBJECTIVE_FUNCTIONS_DICT which maps unique task id strings to black box objective function that we seek to maximize. 
Add additional desired objectives following the examples in this .py file. 
When running LOL-BO and ROBOT, different objective functions can then be used by specifying the unique task id strings using the --task_id argument. 
In addition to a list of xs to be evaluated on the objective, any list of 
additional specific args can be passed in by sepcifying the the argument 
--task_specific_args [arg1,arg2,arg3]

2. your_tasks/your_blackbox_constraints.py 

Defines a dictionary CONSTRAINT_FUNCTIONS_DICT which maps unique id strings to black box constraints that can be applied when running LOL-BO. Add additional desired constraints following the examples in this .py file. 
When running LOL-BO, different constraints can then be added by specifying the unique id strings using the --constraint_function_ids argument. 

3. your_tasks/your_diversity_functions.py 

Defines a dictionary DIVERSITY_FUNCTIONS_DICT which maps unique id strings to diversity functions that can be applied when running ROBOT. Add additional desired diversity functions to the dictionary following the examples in this .py file. 
When running ROBOT, different diversity functions can then be chosen and used by specifying the unique id strings using the --divf_id argument. 

4. Initialization Data 

This should be a csv file with headers x, y 
containing the data used to initilaize optimization 
for both LOL-BO and ROBOT 
(see example in initialization_data/your_init_data.csv ) 
Specify the path to your desired initialization data using the argument
--init_data_path 

5. VAE 

The code is currently set up to use the InfoTransformerVAE 
with the provided example VAE trained on amino acid sequence data from Uniref in uniref_vae/
To swap out the VAE, create files analogous to the following files which define classes with methods that have the same input and output specs and handle encoding and decoding from your VAE as needed.

lolbo/info_transformer_vae_objective.py 

lolbo_scripts/info_transformer_vae_optimization.py

robot/info_transformer_vae_diverse_objective.py

robot_scripts/info_transformer_vae_diverse_optimization.py


## Example Command to run LOL-BO

```Bash
cd lolbo_scripts
```

```Bash
python3 info_transformer_vae_optimization.py --task_id example --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 100 --max_n_oracle_calls 1000 --bsz 10 --dim 1024 --max_string_length 150 - run_lolbo - done 
```

## Example Command to run LOL-BO with One Constraint 
Example runs example optimization task with a max allowed sequence length of 100 

```Bash
python3 info_transformer_vae_optimization.py --task_id example --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 100 --max_n_oracle_calls 1000 --bsz 10 --dim 1024 --max_string_length 150 --constraint_function_ids [length] --constraint_thresholds [100] --constraint_types [max] - run_lolbo - done 
```

## Example Command to run LOL-BO with Multiple Constraints 
Example runs constraining sequences to lengths between 10 and 130, as well as constraining the number of G's to be at least 1 

```Bash
python3 info_transformer_vae_optimization.py --task_id example --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 100 --max_n_oracle_calls 1000 --bsz 10 --dim 1024 --max_string_length 150 --constraint_function_ids [length,length,num_gs] --constraint_thresholds [10,130,1] --constraint_types [min,max,min] - run_lolbo - done 
```

# Example Command to run ROBOT 
```Bash
cd robot_scripts
```

```Bash
python3 info_transformer_vae_diverse_optimization.py --task_id example --divf_id edit_dist --max_n_oracle_calls 1000 --bsz 10 --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 100 --dim 1024 --max_string_length 150 --M 3 --tau 2 - run_robot - done 
```

# Example Commands for Inverse Folding TM Score Optimization
Here, given a target protein structure (pdb file), we seek to find an amino acid sequence that folds into the same structure.
We there use TM score between the folded sequence structure and the target structure as our objective function. 
See our paper Inverse Protein Folding Using Deep Bayesian Optimization (https://arxiv.org/abs/2305.18089) for detials on this task.

Note: When running, don't forget to create initialization for particular target pdb (possibly using ESM-IF by running inverse_folding_oracle/create_esmif_init_data.py) and specify path to data using --init_data_path 

For this task, one task specific argument is required: the target pdb id which is a number 0-23 (i.e. --task_specific_args [0] --> use target structure 0).
Add additional target pdb files to the inverse_folding_oracle/target_pdb_files/ as desired and number them 24, 25, etc. 
Target pdbs 0-23 correspond to the target pdb files used in our paper Inverse Protein Folding Using Deep Bayesian Optimization (https://arxiv.org/abs/2305.18089).

## LOL-BO
```Bash
python3 info_transformer_vae_optimization.py --task_id if_tm_score --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 1000 --max_n_oracle_calls 150000 --bsz 10 --dim 1024 --max_string_length 150 --task_specific_args [0] - run_lolbo - done 
```

## LOL-BO w/ Humanness Constraint 
```Bash
python3 info_transformer_vae_optimization.py --task_id if_tm_score --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 1000 --max_n_oracle_calls 150000 --bsz 10 --dim 1024 --max_string_length 150 --constraint_function_ids [humanness] --constraint_thresholds [0.8] --constraint_types [min] --task_specific_args [3] - run_lolbo - done 
```

## LOL-BO w/ PLDDT Constraint 
```Bash
python3 info_transformer_vae_optimization.py --task_id if_tm_score --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 1000 --max_n_oracle_calls 150000 --bsz 10 --dim 1024 --max_string_length 150 --constraint_function_ids [plddt] --constraint_thresholds [0.8] --constraint_types [min] --task_specific_args [5] - run_lolbo - done 
```

## ROBOT 
```Bash
python3 info_transformer_vae_diverse_optimization.py --task_id if_tm_score --divf_id edit_dist --max_n_oracle_calls 150000 --bsz 10 --track_with_wandb True --wandb_entity $YOUR_WANDB_API_KEY --num_initialization_points 1000 --dim 1024 --max_string_length 150 --M 3 --tau 2 --task_specific_args [2] - run_robot - done 
```

## Example Command to Create Initialization Data with ESM-IF
```Bash
cd inverse_folding_oracle 
```

```Bash
python3 create_esmif_init_data.py --target_pdb_id 0 --bsz 20 --total_num_seqs_generate 1000 
```

# Docker 
nmaus/fold2:latest is public and is the same as the image you'd get my building docker/inverse_fold/Dockerfile 

