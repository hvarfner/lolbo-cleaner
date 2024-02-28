import sys 
sys.path.append("../")
import os 
from inverse_folding_oracle.fold import fold_aa_seq
from inverse_folding_oracle.get_tm import cal_tm_score


from contextlib import contextmanager
import sys, os
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def aa_seq_to_tm_score(
    aa_seq, 
    target_pdb_path,
    esm_model=None,
):
    with suppress_stdout():
        folded_pdb_path = fold_aa_seq(aa_seq, esm_model=esm_model)
        score = cal_tm_score(folded_pdb_path, target_pdb_path)
        os.remove(folded_pdb_path)
    return score 
