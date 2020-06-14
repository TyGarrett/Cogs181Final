import rdkit.Chem as chem
import rdkit.Chem.rdmolfiles as mol
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import io
from Bio.PDB import *

import numpy as np



def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer




data_path = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"



itr = 0
with open(data_path) as file:
    for line in file:

        # skip first line (title)
        if itr == 0:
            itr += 1
            continue

        if itr % 10000 == 0:
            print("Loaded: ", itr)

        if itr >= 100:
            break

        line_vars = line.split('\t')

        peptide_sequence = line_vars[0]
        retention_time = float(line_vars[1].rstrip('\n'))

        # skip len > 20
        if len(peptide_sequence) > 15:
            continue

        m = mol.MolFromFASTA(peptide_sequence)
        AllChem.Compute2DCoords(m)
        pdb = chem.MolToPDBBlock(m)

        parser = PDBParser()
        f = io.StringIO(pdb)
        structure = parser.get_structure('Test', f)

        model = structure[0]
        dist_matrix = calc_dist_matrix(model['A'], model['A'])
        contact_map = dist_matrix < 12.0

        default_mat = np.zeros((15, 15))
        amount_to_pad = 15 - dist_matrix.shape[0]

        new_mat = np.pad(dist_matrix, ((0,amount_to_pad),(0,amount_to_pad)), mode='constant', constant_values=0)
        #print(new_mat)




        itr += 1

