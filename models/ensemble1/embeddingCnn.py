import torch.nn as nn
import torch as torch

class EmbeddingCnn(nn.Module):
    def __init__(self, params):

        batch_size = params['batch_size']
        length_of_sequence = params['length_peptide_sequence']
        input_dim = params['peptide_input_dim']
        number_of_filters = params['number_of_filters']
        kernel_sizes = params['kernel_sizes']
        padding_sizes = params['padding_sizes']
        dropout_rate = params['dropout_rate']

        super(EmbeddingCnn, self).__init__()

        self.embeddings = nn.Embedding(21, 20)


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
                       length_of_sequence,1)
                       #input_dim)
        self.output_dimension = self._get_conv_output(input_shape)
        print(self.output_dimension)

        # define linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.output_dimension, 256)
        )

        # initialize weights
        self._create_weights()

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        #x = x.transpose(1, 2)
        #print(x.size())

        x = self.embeddings(x.long())
        #print(x.size())
        x = x.view(16, 25, 20)
        x = x.transpose(1, 2)

        #print(x.size())

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
        #print(x.size())
        #x = x.transpose(1, 2)
        x = self.embeddings(x.long())
        x = x.view(x.size()[0], x.size()[1], 20)
        x = x.transpose(1, 2)
        x = self.convolution1(x)
        x = self.convolution2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
