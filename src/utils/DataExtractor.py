from src.utils import OneHotEncodingDict
import numpy as np
import math
import torch as torch
from src.utils.peptideDataset import peptideDataset
from torch.utils.data import dataloader
import rdkit.Chem.rdmolfiles as mol
import rdkit.Chem.Descriptors as desc

class DataExtractor:

    def __init__(self, data_path, batch_size, num_data_points, max_length_sequence, backwards=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_data_points = num_data_points
        self.max_length_sequence = max_length_sequence
        self.backwards = backwards

        self.amino_acid_dict = OneHotEncodingDict.getAminoAcidDict()
        self.smiles_dict = OneHotEncodingDict.getSmilesDict()
        self.atom_dict, self.atom_dict_max, self.mol_dict_pos = OneHotEncodingDict.getMolecularDict()

    def load_data_dataloader(self, batch_size):

        x = []
        y = []

        itr = 0
        with open(self.data_path) as file:
            for line in file:

                # skip first line (title)
                if itr == 0:
                    itr += 1
                    continue

                if itr % 10000 == 0:
                    print("Loaded: ", itr)

                if itr >= self.num_data_points:
                    break

                line_vars = line.split('\t')

                peptide_sequence = line_vars[0]
                retention_time = float(line_vars[1].rstrip('\n'))

                # skip len > 20
                if len(peptide_sequence) > self.max_length_sequence:
                    continue

                peptide_matrix = self.peptide_seq_to_onehot_matrix(peptide_sequence)

                x.append(peptide_matrix)
                y.append(retention_time)

                if self.backwards:
                    x.append(peptide_matrix[::-1])
                    y.append(retention_time)

                itr += 1

            print("loaded: {}".format(self.data_path))

            #test train split
            cutoff = math.floor(len(x) * .9)

            X_train = torch.Tensor(x[:cutoff])
            Y_train = torch.Tensor(y[:cutoff])

            X_val = torch.Tensor(x[cutoff:])
            Y_val = torch.Tensor(y[cutoff:])



            my_dataset = torch.utils.data.TensorDataset(X_train, Y_train)  # create your datset
            my_dataloader = dataloader(my_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader

            my_dataset_val = torch.utils.data.TensorDataset(X_val, Y_val)  # create your datset
            my_dataloader_val = dataloader(my_dataset_val, batch_size=batch_size, shuffle=True)  # create your dataloader

            return my_dataloader, my_dataloader_val

    def load_data(self):

        onehot_seq_matrix_list = []
        seq_embedding_list = []
        peptide_seq_list = []
        atom_count_matrix_list = []
        retention_time_list = []

        itr = 0
        with open(self.data_path) as file:
            for line in file:

                # skip first line (title)
                if itr == 0:
                    itr += 1
                    continue

                if itr % 10000 == 0:
                    print("Loaded: ", itr)

                if itr >= self.num_data_points:
                    break

                line_vars = line.split('\t')

                peptide_sequence = line_vars[0]
                retention_time = float(line_vars[1].rstrip('\n'))

                #m = mol.MolFromFASTA("peptide_sequence")

                #print(desc.MolWt(m))
                #print(desc.NumRadicalElectrons(m))
                #print(desc.NumValenceElectrons(m))


                # skip len > 20
                if len(peptide_sequence) > self.max_length_sequence:
                    continue

                onehot_peptide_matrix = self.peptide_seq_to_onehot_matrix(peptide_sequence)
                embedding_matrix = self.peptide_seq_to_embedding_matrix(peptide_sequence)
                atom_count_matrix = self.peptide_seq_to_atom_count_matrix(peptide_sequence)

                onehot_seq_matrix_list.append(onehot_peptide_matrix)
                seq_embedding_list.append(embedding_matrix)
                peptide_seq_list.append(peptide_sequence)
                atom_count_matrix_list.append(atom_count_matrix)
                retention_time_list.append(retention_time)


                if self.backwards:
                    rev_peptide_seq = peptide_sequence[::-1]
                    rev_onehot_peptide_matrix = self.peptide_seq_to_onehot_matrix(rev_peptide_seq)
                    rev_embedding_matrix = self.peptide_seq_to_embedding_matrix(rev_peptide_seq)
                    rev_atom_count_matrix = self.peptide_seq_to_atom_count_matrix(rev_peptide_seq)

                    peptide_seq_list.append(rev_peptide_seq)
                    onehot_seq_matrix_list.append(rev_onehot_peptide_matrix)
                    seq_embedding_list.append(rev_embedding_matrix)
                    atom_count_matrix_list.append(rev_atom_count_matrix)

                    retention_time_list.append(retention_time)

                itr += 1

            print("loaded: {}".format(self.data_path))

            #test train split
            cutoff = math.floor(len(onehot_seq_matrix_list) * .9)

            # train
            onehot_seq_matrix_list_train = torch.Tensor(onehot_seq_matrix_list[ :cutoff])
            peptide_seq_list_train = peptide_seq_list[ :cutoff]
            seq_embedding_list_train = seq_embedding_list[ :cutoff]
            atom_count_matrix_list_train = torch.Tensor(atom_count_matrix_list[ :cutoff])
            retention_time_list_train = torch.Tensor(retention_time_list[ :cutoff])

            train_data = {}
            train_data['onehot_seq_matrix'] = onehot_seq_matrix_list_train
            train_data['peptide_seq_list'] = peptide_seq_list_train
            train_data['seq_embedding_list'] = seq_embedding_list_train
            train_data['atom_count_matrix_list'] = atom_count_matrix_list_train
            train_data['retention_time_list'] = retention_time_list_train

            # validation
            onehot_seq_matrix_list_val = torch.Tensor(onehot_seq_matrix_list[cutoff: ])
            peptide_seq_list_val = peptide_seq_list[cutoff: ]
            seq_embedding_list_val = seq_embedding_list[cutoff: ]
            atom_count_matrix_list_val = torch.Tensor(atom_count_matrix_list[cutoff: ])
            retention_time_list_val = torch.Tensor(retention_time_list[cutoff: ])

            val_data = {}
            val_data['onehot_seq_matrix'] = onehot_seq_matrix_list_val
            val_data['peptide_seq_list'] = peptide_seq_list_val
            val_data['seq_embedding_list'] = seq_embedding_list_val
            val_data['atom_count_matrix_list'] = atom_count_matrix_list_val
            val_data['retention_time_list'] = retention_time_list_val

            train_db = peptideDataset(train_data)
            val_db = peptideDataset(val_data)

            my_dataloader_train = torch.utils.data.DataLoader(train_db,
                                                             batch_size=self.batch_size)  # create your dataloader
            my_dataloader_val = torch.utils.data.DataLoader(val_db,
                                                            batch_size=self.batch_size)  # create your dataloader

            return my_dataloader_train, my_dataloader_val

    def peptide_seq_to_onehot_matrix(self, seq):
        one_hot_matrix = np.zeros((self.max_length_sequence, len(self.amino_acid_dict)))

        mat_itr = 0
        for aa in seq:
            one_hot_matrix[mat_itr][self.amino_acid_dict[aa]] = 1

            mat_itr += 1

        return one_hot_matrix

    def peptide_seq_to_embedding_matrix(self, seq):
        one_hot_matrix = np.zeros((self.max_length_sequence, 1))

        mat_itr = 0
        for aa in seq:
            one_hot_matrix[mat_itr][0] = self.amino_acid_dict[aa] if self.amino_acid_dict[aa] != 0 else 20
            mat_itr += 1

        return one_hot_matrix

    def one_hot_smiles(self, mol, smiles_sequence):
        # 20 peptides x max 50 sequence
        one_hot_matrix = np.zeros((self.max_smiles_seq, len(self.smiles_dict)))

        mat_itr = 0
        for c in smiles_sequence:
            # one hot encode the smiles char
            one_hot_matrix[mat_itr][self.smiles_dict[c]] = 1

        return one_hot_matrix

    def peptide_seq_to_atom_count_matrix(self, seq):
        # 20 peptides x max 50 sequence
        one_hot_matrix = np.zeros((self.max_length_sequence, len(self.atom_dict_max)-1))


        mat_itr = 0
        for aa in seq:
            # one hot encode the smiles char
            one_hot_matrix[mat_itr][self.mol_dict_pos["C"]] = 0 if self.atom_dict_max['C'] == 0 else self.atom_dict[aa]['C']/self.atom_dict_max['C']
            one_hot_matrix[mat_itr][self.mol_dict_pos["H"]] = 0 if self.atom_dict_max['H'] == 0 else self.atom_dict[aa]['H']/self.atom_dict_max['H']
            one_hot_matrix[mat_itr][self.mol_dict_pos["N"]] = 0 if self.atom_dict_max['N'] == 0 else self.atom_dict[aa]['N']/self.atom_dict_max['N']
            one_hot_matrix[mat_itr][self.mol_dict_pos["O"]] = 0 if self.atom_dict_max['O'] == 0 else self.atom_dict[aa]['O']/self.atom_dict_max['O']
            one_hot_matrix[mat_itr][self.mol_dict_pos["S"]] = 0 if self.atom_dict_max['S'] == 0 else self.atom_dict[aa]['S']/self.atom_dict_max['S']
            #one_hot_matrix[mat_itr][self.mol_dict_pos["P"]] = 0 if self.atom_dict_max['P'] == 0 else self.atom_dict[aa]['P']/self.atom_dict_max['P']
            mat_itr += 1

        return one_hot_matrix

