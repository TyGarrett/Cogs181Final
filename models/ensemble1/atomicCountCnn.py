import torch.nn as nn
import torch as torch

class atomCountCNN(nn.Module):
    def __init__(self, params):

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        number_of_filters = params['number_of_filters']
        kernel_sizes = params['kernel_sizes']
        padding_sizes = params['padding_sizes']
        dropout_rate = params['dropout_rate']

        super(atomCountCNN, self).__init__()

        self.convolution1 = nn.Sequential(nn.Conv1d(5,
                                                    256,
                                                    kernel_size=8,
                                                    padding=4),
                                          nn.LeakyReLU(),
                                          nn.Conv1d(256,
                                                    256,
                                                    kernel_size=8,
                                                    padding=4),
                                          nn.LeakyReLU(),
                                          nn.MaxPool1d(2, stride=2)
                                          )

        self.convolution2 = nn.Sequential(nn.Conv1d(256,
                                                    128,
                                                    kernel_size=8,
                                                    padding=4),
                                          nn.LeakyReLU(),
                                          nn.Conv1d(128,
                                                    128,
                                                    kernel_size=8,
                                                    padding=4),
                                          nn.LeakyReLU(),
                                          nn.MaxPool1d(2, stride=2)
                                          )
        self.convolution3 = nn.Sequential(nn.Conv1d(128,
                                                    64,
                                                    kernel_size=8,
                                                    padding=4),
                                          nn.LeakyReLU(),
                                          nn.Conv1d(64,
                                                    64,
                                                    kernel_size=8,
                                                    padding=4),
                                          nn.LeakyReLU()
                                          )

        input_shape = (batch_size,
                       length_of_sequence,
                       5)
        self.output_dimension = self._get_conv_output(input_shape)
        print(self.output_dimension)

        # define linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        # initialize weights
        self._create_weights()

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.convolution1(x)
        x = self.convolution2(x)
        x = self.convolution3(x)

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
        x = self.convolution3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x