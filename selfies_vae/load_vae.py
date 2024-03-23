import sys 
sys.path.append("../")
from selfies_vae.model_positional_unbounded import InfoTransformerVAE 
from selfies_vae.data import SELFIESDataModule, SELFIESDataset
import torch 

# example function to load vae, loads uniref vae 
def load_selfies_vae(
    path_to_vae_statedict,
    dim=1024, # dim//2
    max_string_length=150,
):
    #data_module = SELFIESDataModule(
    #    batch_size=10,
        #k=1,
        #load_data=False,
    #)
    #dataobj = data_module.train
    dataobj = SELFIESDataset()
    vae = InfoTransformerVAE(
        dataset=dataobj, 
        d_model=dim//2,
        kl_factor=0.0001,
    ) 

    # load in state dict of trained model:
    if path_to_vae_statedict:
        state_dict = torch.load(path_to_vae_statedict) 
        vae.load_state_dict(state_dict, strict=True) 
    
    vae = vae.cuda()
    vae = vae.eval()

    # set max string length that VAE can generate
    vae.max_string_length = max_string_length

    return vae, dataobj 
