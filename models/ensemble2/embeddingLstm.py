import torch.nn as nn
import torch as torch

class EmbeddingLstm(nn.Module):
    def __init__(self, params):
        super(EmbeddingLstm, self).__init__()

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        dropout_rate = params['dropout_rate']

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.embeddings = nn.Embedding(21, 20)

        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers)


        input_shape = (batch_size,
                       length_of_sequence,1)
                       #input_dim)
        self.output_dimension = self._get_output_dimension(input_shape)
        print(self.output_dimension)

        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 256),
        )

    def _get_output_dimension(self, shape):
        x = torch.rand(shape)
        x = self.embeddings(x.long())
        x = x.view(x.size()[0], x.size()[1], 20)
        x, _ = self.lstm(x)
        #x = self.hidden2tag2(x)
        x = x.view(x.size()[0], -1)
        output_dimension = x.size()[1]
        return output_dimension

    def forward(self, x):
        x = self.embeddings(x.long())
        x = x.view(x.size()[0], x.size()[1], 20)

        x, _ = self.lstm(x)
        #x = self.hidden2tag2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x