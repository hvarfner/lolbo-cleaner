import torch 
import sys 
sys.path.append("../")
import esm
from inverse_folding_oracle.fold import fold_aa_seq
import os 

def cal_rmsd(c1, c2):
    # given two sets of coordinates, compute the RMSD between them
    # c1 and c2 are (bsz, len, 3, 3) tensors, where len is the number of atoms
    # the 3,3 are just the n, ca, c atoms, stack them together to get the full coordinates
    #print(c1.shape)
    # using contiguous() to make sure the tensor is stored in a contiguous block of memory
    c1 = c1.contiguous().view(c1.shape[0]*3, 3)
    c2 = c2.contiguous().view(c2.shape[0]*3, 3)
    #print(c1.shape)

    device = c1.device
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        assert False, "SVD failed"
        # return torch.tensor([20.0], device=device)#, False
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    return msd.sqrt()#, True


def cal_unnormalized_rmsd(inverse_folded_pdb, target_pdb):
    # Load the target structure and the inverse folded structure
    structure_target = esm.inverse_folding.util.load_structure(target_pdb, "A")
    coords_target, _ = esm.inverse_folding.util.extract_coords_from_structure(structure_target)

    structure_inverse_folded = esm.inverse_folding.util.load_structure(inverse_folded_pdb, "A")
    coords_inverse_folded, _ = esm.inverse_folding.util.extract_coords_from_structure(structure_inverse_folded)

    rmsd = cal_rmsd(torch.tensor(coords_target), torch.tensor(coords_inverse_folded))

    return rmsd.item()


def aa_seq_to_rmsd_score(
    aa_seq, 
    target_pdb_path,
    esm_model=None,
):
    folded_pdb_path = fold_aa_seq(aa_seq, esm_model=esm_model)
    rmsd = cal_unnormalized_rmsd(folded_pdb_path, target_pdb_path)
    os.remove(folded_pdb_path)
    return rmsd 