import esm 
device = "cuda:0"
def score_sequences(seq_list, target_pdb, inv_model, alphabet, device=device):

    inv_model = inv_model.to(device)

    structure = esm.inverse_folding.util.load_structure(target_pdb, "A")
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

    ll, _ = esm.inverse_folding.util.score_sequence(inv_model, alphabet, coords, seq_list)

    perplexity = np.exp(-np.array(ll))

    return perplexity
