def getAminoAcidDict():
    """    amino_acid_dict = {}
        amino_acid_dict["A"] = 0
        amino_acid_dict["R"] = 1
        amino_acid_dict["N"] = 2
        amino_acid_dict["D"] = 3
        amino_acid_dict["C"] = 4
        amino_acid_dict["Q"] = 5
        amino_acid_dict["E"] = 6
        amino_acid_dict["G"] = 7
        amino_acid_dict["H"] = 8
        amino_acid_dict["I"] = 9
        amino_acid_dict["L"] = 10
        amino_acid_dict["K"] = 11
        amino_acid_dict["M"] = 12
        amino_acid_dict["F"] = 13
        amino_acid_dict["P"] = 14
        amino_acid_dict["S"] = 15
        amino_acid_dict["T"] = 16
        amino_acid_dict["W"] = 17
        amino_acid_dict["Y"] = 18
        amino_acid_dict["V"] = 19

        return amino_acid_dict
    """

    dict_aa={
         "K": 0,
         "R": 1,
         "P": 2,
         "T": 3,
         "N": 4,
         "A": 5,
         "Q": 6,
         "V": 7,
         "S": 8,
         "G": 9,
         "I": 10,
         "L": 11,
         "C": 12,
         "M": 13,
         "H": 14,
         "F": 15,
         "Y": 16,
         "W": 17,
         "E": 18,
         "D": 19}

    return dict_aa

def getSmilesDict():
    smiles_dict = {}
    smiles_dict['H'] = 0
    smiles_dict['O'] = 1
    smiles_dict['C'] = 2
    smiles_dict['N'] = 3
    smiles_dict['S'] = 4
    smiles_dict['['] = 5
    smiles_dict[']'] = 6
    smiles_dict['@'] = 7
    smiles_dict['('] = 8
    smiles_dict[')'] = 9
    smiles_dict['='] = 10
    smiles_dict['1'] = 11
    smiles_dict['c'] = 12
    smiles_dict['n'] = 13
    smiles_dict['2'] = 14
    return smiles_dict

def getMolecularDict():
    mol_dict = {}
    mol_dict["A"] = {"C": 1, "H": 2, "N": 0, "O": 0, "S": 0, "P": 0}
    mol_dict["R"] = {"C": 4, "H": 9, "N": 3, "O": 0, "S": 0, "P": 0}
    mol_dict["N"] = {"C": 2, "H": 3, "N": 1, "O": 1, "S": 0, "P": 0}
    mol_dict["D"] = {"C": 2, "H": 2, "N": 0, "O": 2, "S": 0, "P": 0}
    mol_dict["C"] = {"C": 1, "H": 2, "N": 0, "O": 0, "S": 1, "P": 0}
    mol_dict["Q"] = {"C": 3, "H": 5, "N": 1, "O": 1, "S": 0, "P": 0}
    mol_dict["E"] = {"C": 3, "H": 4, "N": 0, "O": 2, "S": 0, "P": 0}
    mol_dict["G"] = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0, "P": 0}
    mol_dict["H"] = {"C": 4, "H": 4, "N": 2, "O": 0, "S": 0, "P": 0}
    mol_dict["I"] = {"C": 4, "H": 8, "N": 0, "O": 0, "S": 0, "P": 0}
    mol_dict["L"] = {"C": 4, "H": 8, "N": 0, "O": 0, "S": 0, "P": 0}

    mol_dict["K"] = {"C": 4, "H": 9, "N": 1, "O": 0, "S": 0, "P": 0}
    mol_dict["M"] = {"C": 3, "H": 6, "N": 0, "O": 0, "S": 1, "P": 0}
    mol_dict["F"] = {"C": 7, "H": 6, "N": 0, "O": 0, "S": 0, "P": 0}
    mol_dict["P"] = {"C": 3, "H": 4, "N": 0, "O": 0, "S": 0, "P": 0}
    mol_dict["S"] = {"C": 1, "H": 2, "N": 0, "O": 1, "S": 0, "P": 0}
    mol_dict["T"] = {"C": 2, "H": 4, "N": 0, "O": 1, "S": 0, "P": 0}
    mol_dict["W"] = {"C": 9, "H": 7, "N": 1, "O": 0, "S": 0, "P": 0}
    mol_dict["Y"] = {"C": 7, "H": 6, "N": 0, "O": 1, "S": 0, "P": 0}
    mol_dict["V"] = {"C": 3, "H": 6, "N": 0, "O": 0, "S": 0, "P": 0}
    mol_dict["X"] = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0, "P": 0}


    max_val_dic = {"C": 9, "H": 9, "N": 3, "O": 2, "S": 1, "P": 0}
    mol_dict_pos = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}
    return mol_dict, max_val_dic, mol_dict_pos

"""def getMolecularDict():
    mol_dict = {}
    mol_dict["A"] = {"C": 3, "H": 7, "N": 1, "O": 2, "S": 0, "P": 0}
    mol_dict["R"] = {"C": 6, "H": 14, "N": 4, "O": 2, "S": 0, "P": 0}
    mol_dict["N"] = {"C": 4, "H": 8, "N": 2, "O": 3, "S": 0, "P": 0}
    mol_dict["D"] = {"C": 4, "H": 7, "N": 1, "O": 4, "S": 0, "P": 0}
    mol_dict["C"] = {"C": 3, "H": 7, "N": 2, "O": 2, "S": 1, "P": 0}
    mol_dict["Q"] = {"C": 5, "H": 10, "N": 2, "O": 3, "S": 0, "P": 0}
    mol_dict["E"] = {"C": 5, "H": 9, "N": 1, "O": 4, "S": 0, "P": 0}
    mol_dict["G"] = {"C": 2, "H": 5, "N": 1, "O": 2, "S": 0, "P": 0}
    mol_dict["H"] = {"C": 6, "H": 9, "N": 3, "O": 2, "S": 0, "P": 0}
    mol_dict["I"] = {"C": 6, "H": 13, "N": 1, "O": 2, "S": 0, "P": 0}
    mol_dict["L"] = {"C": 6, "H": 13, "N": 1, "O": 2, "S": 0, "P": 0}
    mol_dict["K"] = {"C": 6, "H": 14, "N": 2, "O": 2, "S": 0, "P": 0}
    mol_dict["M"] = {"C": 5, "H": 11, "N": 1, "O": 2, "S": 1, "P": 0}
    mol_dict["F"] = {"C": 9, "H": 11, "N": 1, "O": 2, "S": 0, "P": 0}
    mol_dict["P"] = {"C": 5, "H": 9, "N": 1, "O": 2, "S": 0, "P": 0}
    mol_dict["S"] = {"C": 3, "H": 7, "N": 1, "O": 3, "S": 0, "P": 0}
    mol_dict["T"] = {"C": 4, "H": 9, "N": 1, "O": 3, "S": 0, "P": 0}
    mol_dict["W"] = {"C": 11, "H": 12, "N": 2, "O": 2, "S": 0, "P": 0}
    mol_dict["Y"] = {"C": 9, "H": 11, "N": 1, "O": 3, "S": 0, "P": 0}
    mol_dict["V"] = {"C": 5, "H": 11, "N": 1, "O": 2, "S": 0, "P": 0}

    max_val_dic = {"C": 11, "H": 14, "N": 3, "O": 4, "S": 1, "P": 0}
    mol_dict_pos = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}
    return mol_dict, max_val_dic, mol_dict_pos"""