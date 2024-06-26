import torch
import math
from torch.utils.data import TensorDataset, DataLoader
from lolbo.utils.bo_utils.ppgpr import ZGPModel
from lolbo.utils.bo_utils.zrbf import ZRBFKernel
from gpytorch.distributions import MultivariateNormal


def update_models_end_to_end_unconstrained(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    vae_learning_rate,
    gp_learning_rate,
    num_update_epochs,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    objective.vae.train()
    model.train() 
    optimize_list = [
        {'params': objective.vae.parameters(), 'lr': vae_learning_rate},
        {'params': model.parameters(), 'lr': gp_learning_rate} 
    ]
    optimizer = torch.optim.Adam(optimize_list)
    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560/max_string_length)) 
    num_batches = math.ceil(len(train_x) / bsz)
    for _ in range(num_update_epochs):
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            z, vae_loss = objective.vae_forward(batch_list)
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float() 
            pred = model(z)
            surr_loss = -mll(pred, batch_y.cuda())
            # add losses and back prop 
            loss = vae_loss + surr_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
            optimizer.step()
    objective.vae.eval()
    model.eval()

    return objective, model


def update_models_end_to_end_with_constraints(
    train_x,
    train_y_scores,
    objective,
    model,
    mll,
    vae_learning_rate,
    gp_learning_rate,
    num_update_epochs,
    train_c_scores=None,
    c_models=[],
    c_mlls=[],
    freeze_vae: bool = False,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    SELFIES VAE interface
    '''
    objective.vae.train()
    model.train() 
    optimize_list = [
        {'params': objective.vae.parameters(), 'lr': vae_learning_rate},
        {'params': model.parameters(), 'lr': gp_learning_rate} 
    ]
    if train_c_scores is not None:
        for c_model in c_models:
            c_model.train() 
            optimize_list.append({f"params": c_model.parameters(), 'lr': gp_learning_rate})
    optimizer = torch.optim.Adam(optimize_list) 

    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560 * 2/max_string_length)) 
    num_batches = math.ceil(len(train_x) / bsz)

    # This is the new kernel, checking for it here to do the alternative training
    train_on_z = isinstance(model.covar_module.base_kernel, ZRBFKernel)
    for ep in range(num_update_epochs):
        
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float() 
            # This is also only for the new training loop - please disregard
            if train_on_z:
                _, vae_loss, z_mu, z_sigma = objective.vae_forward(batch_list, return_mu_sigma=train_on_z)
                if freeze_vae:
                    z_mu = z_mu.detach() 
                    z_sigma = z_sigma.detach()
                z_full = torch.cat((z_mu, z_sigma ** 2), dim=-1)
                pred = model(z_full)
                
            else:
                z, vae_loss, z_mu, z_sigma, token_loss, string_loss = objective.vae_forward(batch_list, return_losses=True)
                #print("Inside e2e", token_loss, string_loss)
                pred = model(z)
            
            surr_loss = -mll(pred, batch_y.cuda())
            
            if train_c_scores is not None:
                batch_c = train_c_scores[start_idx:stop_idx]
                for ix, c_model in enumerate(c_models):
                    batch_c_ix = batch_c[:,ix] 
                    c_pred_ix = c_model(z) 
                    loss_cmodel_ix = -c_mlls[ix](c_pred_ix, batch_c_ix.cuda())
                    surr_loss = surr_loss + loss_cmodel_ix

            # add losses and back prop         
            #print(surr_loss, vae_loss)
            loss = surr_loss + vae_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
            optimizer.step()
            
    objective.vae.eval()
    model.eval()
    if train_c_scores is not None:
        for c_model in c_models:
            c_model.eval() 

    return objective, model


def update_surr_model(
    model,
    mll,
    gp_learning_rate,
    train_z,
    train_y,
    n_epochs
):
    model = model.train() 
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': gp_learning_rate}])
    train_bsz = min(len(train_y),128)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()

    return model


def update_constraint_surr_models(
    c_models,
    c_mlls,
    gp_learning_rate,
    train_z,
    train_c,
    n_epochs,
):
    updated_c_models = []
    for ix, c_model in enumerate(c_models):
        updated_model = update_surr_model(
            c_model,
            c_mlls[ix],
            gp_learning_rate,
            train_z,
            train_c[:,ix],
            n_epochs
        )
        updated_c_models.append(updated_model)
    
    return updated_c_models

