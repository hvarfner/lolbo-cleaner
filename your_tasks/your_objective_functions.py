''' Define your objecive function(s) here
Note: All code assumes we seek to maximize f(x)
If you want to instead MINIMIZE the objecitve, multiple scores by -1 in 
your query_black_box() method 
''' 

import sys 
sys.path.append("../")
import numpy as np 
import os 
try:
    # dependencies for inverse fold oracle 
    from inverse_folding_oracle.aa_seq_to_tm_score import aa_seq_to_tm_score
    from transformers import EsmForProteinFolding
except:
    print("Inverse Folding Oracle not Availalbe in this environment")
    print("Please use container in docker/inverse_fold/Dockerfile to run IF optimization\n")
from lolbo.utils.mol_utils import selfies_to_desired_scores

class ObjectiveFunction:
    ''' Objective function f, we seek x that MAXIMIZE f(x)'''
    def __init__(self,):
        pass
    
    def __call__(self, x_list):
        ''' Input 
                x_list: 
                    a LIST of input space items from the origianl input 
                    search space (i.e. list of aa seqs)
            Output 
                scores_list: 
                    a LIST of float values obtained by evaluating your 
                    objective function f on each x in x_list
                    or np.nan in the wherever x is an invalid input 
        '''
        return self.query_black_box(x_list)


    def query_black_box(self, x_list):
        ''' Input 
                x_list: a list of input space items x_list
            Output 
                scores_list: 
                    a LIST of float values obtained by evaluating your 
                    objective function f on each x in x_list
                    or np.nan in the wherever x is an invalid input 
        '''
        raise NotImplementedError("Must implement method query_black_box() for the black box objective")


class ExampleObjective(ObjectiveFunction):
    ''' Example objective funciton length of the input space items
        This is just a dummy example where the objective is the 
        avg number of A and M's in the sequence 
    ''' 
    def __init__(self,):
        super().__init__()

    def query_black_box(self, x_list):
        scores_list = []
        for x in x_list:
            if type(x) != str:
                score = np.nan 
            elif len(x) == 0:
                score = 0 
            else:
                score = 0 
                for char in x: 
                    if char in ["A", "M"]:
                        score += 1
                score = score/len(x)
            scores_list.append(score)

        return scores_list 
    

class GuacamolObjective(ObjectiveFunction):
    ''' Example objective funciton length of the input space items
        This is just a dummy example where the objective is the 
        avg number of A and M's in the sequence 
    ''' 
    def __init__(self, guac_name):
        super().__init__()
        self.guac_name = guac_name

    def query_black_box(self, x_list):
        return selfies_to_desired_scores(x_list, self.guac_name).tolist()
 


class MentholObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("med1")

        
class SildenafilObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("pdop")

        
class PerindoprilRingsObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("adip")

        
class OsimertinibObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("rano")

        
class AmlodipineRingsObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("osmb")

        
class SitagliptinObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("siga")

        
class ZaleplonObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("zale")

        
class ValsartanSmartsObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("valt")

        
class DecorationHopObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("med2")

        
class ScaffoldHopObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("dhop")

        
class RanolazineMpoObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("shop")

        
class FexofenadineObjective(GuacamolObjective):
    def __init__(self):
        super().__init__("fexo")

        

class InverseFoldTMScoreObjective(ObjectiveFunction):
    ''' Objective function for optimizing the TM Score between the input sequence 
        (after folding with ESM), and the target protein structure
    ''' 
    def __init__( 
        self,
        target_pdb_id, # id number for the target structure 
    ):
        self.esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.esm_model = self.esm_model.eval() 
        self.esm_model = self.esm_model.cuda()
        self.target_pdb_path = f"../inverse_folding_oracle/target_pdb_files/target_structure_{target_pdb_id}.pdb"
        assert os.path.exists(self.target_pdb_path)
        super().__init__()

    def query_black_box(self, x_list):
        scores_list = [] 
        for x in x_list:
            score = aa_seq_to_tm_score(
                aa_seq=x, 
                target_pdb_path=self.target_pdb_path,
                esm_model=self.esm_model,
            ) 
            scores_list.append(score)
        return scores_list 


'''Objective functions with unique string identifiers 
identifiers can be passed in when running LOL-BO or ROBOT with --task_id arg
whcih specifies which diversity function to use 
--task_specific_args can be used to specify a list of args passed into the init of 
any of these objectives when they are initialized 
'''
OBJECTIVE_FUNCTIONS_DICT = {
    "example":ExampleObjective,
    "if_tm_score":InverseFoldTMScoreObjective,
    "med1": MentholObjective,
    "pdop": SildenafilObjective,
    "adip": PerindoprilRingsObjective,
    "rano": OsimertinibObjective,
    "osmb": AmlodipineRingsObjective,
    "siga": SitagliptinObjective,
    "zale": ZaleplonObjective,
    "valt": ValsartanSmartsObjective,
    "med2": DecorationHopObjective,
    "dhop": ScaffoldHopObjective,
    "shop": RanolazineMpoObjective,
    "fexo": FexofenadineObjective,

}
