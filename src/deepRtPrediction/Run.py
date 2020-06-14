from src.utils.DataExtractor import DataExtractor
from models.CNN import PeptideCNN
from models.lstm import peptideLstm
from models.ensemble2.ensemble import peptideEnsemble

from src.deepRtPrediction.Train import TrainModel
import matplotlib.pyplot as plt

import os


class Run():

    def __init__(self, params):
        self.params = params
        self.name = params['name']
        self.num_data_points = 500000
        self.run()


    def run(self):

        params = self.params
        data_path = params['data_path']
        max_seq_length = params['length_peptide_sequence']
        batch_size = params['batch_size']
        isBackwards = params['backwards']

        data_loader = DataExtractor(data_path, batch_size, self.num_data_points, max_length_sequence=max_seq_length, backwards=isBackwards)
        data_loader_train, data_loader_val = data_loader.load_data()

        model = peptideEnsemble(params)
        print(model)

        results = TrainModel(model, params, data_loader_train, data_loader_val)
        results.train()

        self.save_results(results)


    def save_results(self, results):

        path ='Output\\{}'.format(self.name)

        if not os.path.exists(path):
            os.mkdir(path);

        plt.plot(results.avg_losses)
        plt.ylabel('training loss')
        plt.savefig('Output\\{}\\training_loss.png'.format(self.name))
        plt.clf()

        plt.plot(results.r2)
        plt.ylabel('r2')
        plt.savefig('Output\\{}\\r2.png'.format(self.name))
        plt.clf()

        plt.plot(results.val_mse)
        plt.ylabel('val_rmses')
        plt.savefig('Output\\{}\\val_rmses.png'.format(self.name))
        plt.clf()

        f = open("Output\\{}\\hyperParams.txt".format(self.name), "w+")
        f.write(str(self.params))
        f.close()

        f = open("Output\\{}\\Results.txt".format(self.name), "w+")
        f.write("results.avg_losses: {} \n".format(results.avg_losses))
        f.write("results.r2: {} \n".format(results.r2))
        f.write("results.val_rmses: {} \n".format(results.val_mse))

        min_val = min(results.avg_losses)
        epoch_min_val = results.avg_losses.index(min_val)
        f.write("min_avg_losses: {} | Epoch: {} \n".format(min_val, epoch_min_val))

        min_val = max(results.r2)
        epoch_min_val = results.r2.index(min_val)
        f.write("r2: {} | Epoch: {} \n".format(min_val, epoch_min_val))

        min_val = min(results.val_mse)
        epoch_min_val = results.val_mse.index(min_val)
        f.write("val_rmses: {} | Epoch: {} \n".format(min_val, epoch_min_val))

        f.close()

