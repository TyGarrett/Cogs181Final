import torch.nn as nn
import torch as torch

class peptideLstm(nn.Module):
    def __init__(self, params):
        super(peptideLstm, self).__init__()

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        dropout_rate = params['dropout_rate']

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers)

        input_shape = (batch_size,
                       length_of_sequence,
                       input_dim)
        self.output_dimension = self._get_output_dimension(input_shape)
        print(self.output_dimension)

        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256)
        )

    def _get_output_dimension(self, shape):
        x = torch.rand(shape)
        x, _ = self.lstm(x)
        x = x.view(x.size()[0], -1)
        output_dimension = x.size()[1]
        return output_dimension

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class PeptideCNN(nn.Module):
    def __init__(self, params):

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        number_of_filters = params['number_of_filters']
        kernel_sizes = params['kernel_sizes']
        padding_sizes = params['padding_sizes']
        dropout_rate = params['dropout_rate']

        super(PeptideCNN, self).__init__()

        self.convolution1 = nn.Sequential(nn.Conv1d(input_dim,
                                                    number_of_filters,
                                                    kernel_size=kernel_sizes[0],
                                                    padding=padding_sizes[0]),
                                          nn.ReLU()
                                          )

        self.convolution2 = nn.Sequential(nn.Conv1d(number_of_filters,
                                                    number_of_filters,
                                                    kernel_size=kernel_sizes[1],
                                                    padding=padding_sizes[1]),
                                          nn.ReLU()
                                          )

        input_shape = (batch_size,
                       length_of_sequence,
                       input_dim)
        self.output_dimension = self._get_conv_output(input_shape)
        print(self.output_dimension)

        # define linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 556),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(556, 256),
        )

        # initialize weights
        self._create_weights()

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.convolution1(x)
        x = self.convolution2(x)
        x = x.view(x.size(0), -1)
        output_dimension = x.size()[1]
        return output_dimension

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.convolution1(x)
        x = self.convolution2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class peptideEnsemble(nn.Module):
    def __init__(self, params):
        super(peptideEnsemble, self).__init__()

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        dropout_rate = params['dropout_rate']

        self.lstm = peptideLstm(params)
        self.cnn = PeptideCNN(params)

        input_shape = (batch_size,
                       length_of_sequence,
                       input_dim)
        self.output_dimension = self._get_output_dimension(input_shape)
        print(self.output_dimension)

        # define linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )


    def _get_output_dimension(self, shape):
        x = torch.rand(shape)

        x1 = self.lstm(x)
        x2 = self.cnn(x)
        x = torch.cat((x1, x2), dim=1)

        output_dimension = x.size()[1]
        return output_dimension

    def forward(self, x):
        x1 = self.lstm(x)
        x2 = self.cnn(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x