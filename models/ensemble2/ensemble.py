import torch.nn as nn
import torch as torch

from models.ensemble2.embeddingCnn1 import *

class peptideEnsemble(nn.Module):
    def __init__(self, params):
        super(peptideEnsemble, self).__init__()

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        dropout_rate = params['dropout_rate']

        self.embeddings = nn.Embedding(21, 20)

        self.cnn = EmbeddingCnn1(params)
        self.cnn2 = EmbeddingCnn2(params)
        self.cnn3 = EmbeddingCnn3(params)

        input_shape = (batch_size,
                       length_of_sequence,
                       1)
        atom_shape = (batch_size,
                       length_of_sequence,5)
                       #input_dim)
        self.output_dimension = self._get_output_dimension(input_shape, atom_shape)
        print(self.output_dimension)

        # define linear layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
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
        #x = self.embeddings(x.long())
        #print(x.size())
        #x = x.view(x.size()[0], 25, 20)

        #x = self.embeddings(x.long())
        #print(x.size())
        #x = x.view(16, 25, 20)

        x1 = self.cnn(x)
        #x2 = self.cnn2(x)
        #x3 = self.cnn3(x)
        #x = torch.cat((x1, x2, x3), dim=1)

        output_dimension = x.size()[1]
        return output_dimension

    def forward(self, x, atom_matrix):
        #x = self.embeddings(x.long())
        #print(x.size())
        #x = x.view(x.size()[0], 25, 20)

        x = self.cnn(x)
        #x2 = self.cnn2(x)
        #x3 = self.cnn3(x)
        #x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x