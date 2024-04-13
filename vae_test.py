#-----------------------------------------------------------
# THIS IS ADDED
import numpy as np
import torch
import pandas as pd
from lolbo.info_transformer_vae_objective import InfoTransformerVAEObjective
from selfies_vae.load_vae import load_selfies_vae

def load_data(init_data_path: str, seed: int = 1):
    import pandas as pd
    df = pd.read_csv(init_data_path)

    gen = np.random.default_rng(seed=seed)
    point_indices = gen.choice(len(df), size=len(df), replace=False)
    x = df.loc[point_indices, "x"].to_list() 
    y = df.loc[point_indices, "y"].to_numpy() 
    y = torch.from_numpy(y).float().unsqueeze(-1) 
    return x, y

if __name__ == "__main__":
    INIT_DATA_PATH = "selfies_vae/init/init_rano.csv"
    obj = InfoTransformerVAEObjective(
        task_id="rano",
        path_to_vae_statedict="selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        max_string_length=1024, # the default in LOL-BO
        dim=256,
        vae_load_function=load_selfies_vae,
        num_calls=-1, # Modularized how to load the VAE
        task_specific_args=[],
        constraint_function_ids=[],
        constraint_thresholds=[],
        constraint_types=[],
    )

    train_x, train_y  = load_data(init_data_path=INIT_DATA_PATH) 

    batch_size = 16
    num_batches = np.ceil(len(train_x) / batch_size).astype (int)
    token_recon_per_batch = torch.zeros(num_batches)
    string_recon_per_batch = torch.zeros(num_batches)
    
    for batch_ix in range(num_batches):
        if batch_ix % 10 == 0:
            print(f"{batch_ix}  / {num_batches}, "
                f"Token recon: {token_recon_per_batch[:batch_ix].mean()} --- "
                f"String recon: {string_recon_per_batch[:batch_ix].mean()}"
            )
        start_idx, stop_idx = batch_ix * batch_size, (batch_ix+1) * batch_size
        batch_list = train_x[start_idx:stop_idx]
        loss_dict = obj.vae_forward(batch_list, return_dict=True)
        recon_token, recon_string = loss_dict["recon_token_acc"], loss_dict["recon_string_acc"]
        token_recon_per_batch[batch_ix] = recon_token
        string_recon_per_batch[batch_ix] = recon_string

    print("Final result:")
    print(f"{batch_ix}  / {num_batches}, "
        f"Token recon: {token_recon_per_batch.mean()} --- "
        f"String recon: {string_recon_per_batch.mean()}"
    )