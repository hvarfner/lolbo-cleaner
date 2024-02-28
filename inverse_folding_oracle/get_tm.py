import re
import subprocess
import numpy as np 

def cal_tm_score(folded_pdb, target_pdb):
    # Call the executable file in Bash and capture its output
    command = ["../inverse_folding_oracle/TMalign", folded_pdb, target_pdb]
    output = subprocess.check_output(command)
    # Extract the TM-score value using regular expressions
    tm_score_regex = r"TM-score= ([\d\.]+)"
    # tm_score_match = re.search(tm_score_regex, output.decode("utf-8"))
    tm_score_matches = re.findall(tm_score_regex, output.decode("utf-8")) 
    if tm_score_matches: 
        tm_score1 = float(tm_score_matches[0])
        tm_score2 = float(tm_score_matches[1])
        tm_score = min(tm_score1, tm_score2)
    else:
        # print("TM-score not found, there might be an error with the pdb file.")
        tm_score = np.nan 

    # if tm_score_match:
    #     tm_score = float(tm_score_match.group(1))
    # else:
    #     print("TM-score not found, there might be an error with the pdb file.")
    #     tm_score = np.nan 
    return tm_score


def cal_tm_score_and_normalized_rmsd(folded_pdb, target_pdb):
    # Call the executable file in Bash and capture its output
    command = ["../inverse_folding_oracle/TMalign", folded_pdb, target_pdb]
    output = subprocess.check_output(command)

    decoded_output = output.decode("utf-8")
    # Extract normalized RMSD value using regular expressions
    rmsd_regex = r"RMSD=\s+([\d.]+)"
    rmsd_matches = re.findall(rmsd_regex, decoded_output)

    if rmsd_matches:
        rmsd = float(rmsd_matches[0])
    else:
        # print("RMSD not found, there might be an error with the pdb file.")
        rmsd = np.nan
    
    # Extract the TM-score value using regular expressions
    tm_score_regex = r"TM-score= ([\d\.]+)"
    tm_score_matches = re.findall(tm_score_regex, decoded_output) 
    if tm_score_matches: 
        tm_score1 = float(tm_score_matches[0])
        tm_score2 = float(tm_score_matches[1])
        tm_score = min(tm_score1, tm_score2)
    else:
        # print("TM-score not found, there might be an error with the pdb file.")
        tm_score = np.nan
    return tm_score, rmsd


if __name__ == "__main__":
    score = cal_tm_score(folded_pdb="target_pdb_files/17_bp_sh3.ent", target_pdb="target_pdb_files/33_bp_sh3.ent")
    print(f"Score is: {score}")
    score = cal_tm_score(folded_pdb="target_pdb_files/17_bp_sh3.ent", target_pdb="target_pdb_files/17_bp_sh3.ent")
    print(f"Score is: {score}")
