import pymol2
import glob 

def convert_pdb_to_cif(pdb_file):
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file,'myprotein')
        pymol.cmd.save(pdb_file.replace('.pdb', '.cif'), selection='myprotein')

def convert_ent_to_cif(pdb_file):
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file,'myprotein')
        pymol.cmd.save(pdb_file.replace('.ent', '.cif'), selection='myprotein')

def convert_cif_to_pdb(pdb_file):
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file,'myprotein')
        pymol.cmd.save(pdb_file.replace('.cif', '.pdb'), selection='myprotein')


if __name__ == "__main__":
    # conda activate pymol 
    # 170_44
    # 240_16
    # 260_9
    # 270_2
    # 270_3
    # 270_14 
    # 300_16
    # 300_28 
    pdb_files = glob.glob("target_pdb_files/sample*.pdb")
    for pdb_file in pdb_files:
        convert_pdb_to_cif(pdb_file)
