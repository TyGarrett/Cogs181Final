import torch.nn as nn
import torch as torch

from models.ensemble1.embeddingLstm import EmbeddingLstm
from models.ensemble1.embeddingCnn import EmbeddingCnn
from models.ensemble1.atomicCountCnn import atomCountCNN

class peptideEnsemble(nn.Module):
    def __init__(self, params):
        super(peptideEnsemble, self).__init__()

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        dropout_rate = params['dropout_rate']

        self.lstm = EmbeddingLstm(params)
        self.cnn = EmbeddingCnn(params)
        self.atomCnn = atomCountCNN(params)

        input_shape = (batch_size,
                       length_of_sequence,1)
                       #input_dim)
        atom_shape = (batch_size,
                       length_of_sequence,5)
                       #input_dim)
        self.output_dimension = self._get_output_dimension(input_shape, atom_shape)
        print(self.output_dimension)

        # define linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def _get_output_dimension(self, shape, atomShape):
        x = torch.rand(shape)
        atom_matrix = torch.rand(atomShape)

        x1 = self.lstm(x)
        x2 = self.cnn(x)
        x3 = self.atomCnn(atom_matrix)
        x = torch.cat((x1, x2, x3), dim=1)

        output_dimension = x.size()[1]
        return output_dimension

    def forward(self, x, atom_matrix):
        x1 = self.lstm(x)
        x2 = self.cnn(x)
        x3 = self.atomCnn(atom_matrix)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x