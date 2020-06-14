from torch.utils.data import Dataset, DataLoader
import torch as torch
import numpy as np


class peptideDataset(Dataset):

    def __init__(self, data):

        self.one_hot_peptide_seq =  data['onehot_seq_matrix']
        self.embedding_matrix = data['seq_embedding_list']
        self.peptide_sequence = data['peptide_seq_list']
        self.atom_count_mat = data['atom_count_matrix_list']

        self.retention_time = data['retention_time_list']

    def __len__(self):
        return len(self.one_hot_peptide_seq)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mat = self.one_hot_peptide_seq[idx]
        emb = self.embedding_matrix[idx]
        seq = self.peptide_sequence[idx]
        atom_mat = self.atom_count_mat[idx]

        rt = self.retention_time[idx]
        sample = {'peptide_matrix': mat, 'sequence': seq, 'atom_matrix': atom_mat,
                  'embedding_matrix': emb, 'retention_time': rt}

        return sample
