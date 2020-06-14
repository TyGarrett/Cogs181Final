import math
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


class TrainModel():

    def __init__(self, model, params, data_loader_train, data_loader_val):

        self.model = model
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val

        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']

        self.momentum = params['momentum']
        self.nesterov = params['nesterov']

        # SetUp GPU
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Loss Function / Optimizer
        self.loss_func = nn.MSELoss()
        self.opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=self.nesterov)

        self.val_percent_diff = []
        self.val_mse = []
        self.avg_losses = []  # Avg. losses.
        self.r2 = []

    def train(self):

        self.model.train()
        for epoch in range(self.epochs):  # Loop over the dataset multiple times.
            running_loss = 0.0  # Initialize running loss.
            for i, data in enumerate(self.data_loader_train, 0):
                # Get the inputs.
                inputs = data['peptide_matrix'].to(self.device)
                emb = data['embedding_matrix'].to(self.device)
                seq = data['sequence']
                atom_mat = data['atom_matrix'].to(self.device)
                labels = data['retention_time'].to(self.device)

                # Zero the parameter gradients.
                self.opt.zero_grad()

                # Forward step.
                outputs = self.model(emb, atom_mat).reshape(-1)

                # Calculate the training loss
                loss = self.loss_func(outputs, labels)

                # Backward step.
                loss.backward()

                # Optimization step (update the parameters).
                self.opt.step()

                # Print statistics.
                running_loss += loss.item()

            running_loss = running_loss / i
            percent_diff, mse, r_2 = self.validate()

            print("Epoch: {} | Train Loss: {} | percent_diff: {} | val mse: {} | Val r_2: {}".format(
                epoch, running_loss, percent_diff, mse, r_2))

            self.val_percent_diff.append(percent_diff)
            self.val_mse.append(mse)
            self.r2.append(r_2)
            self.avg_losses.append(running_loss)


    def validate(self):

        predictions = []
        targets = []

        running_ave_itr = 0
        with torch.no_grad():
            for data in self.data_loader_val:
                inputs = data['peptide_matrix'].to(self.device)
                emb = data['embedding_matrix'].to(self.device)
                seq = data['sequence']
                atom_mat = data['atom_matrix'].to(self.device)
                labels = data['retention_time'].to(self.device)

                outputs = self.model(emb, atom_mat).reshape(-1)

                labels_cpu=labels.cpu()
                outputs_cpu=outputs.cpu()

                targets.extend(labels_cpu)
                predictions.extend(outputs_cpu)


        pearsonr_ave = pearsonr(targets, predictions)[0]

        print(outputs_cpu)
        print(labels_cpu)

        plist = []
        for i in range(len(targets)):
            t = targets[i]
            p = predictions[i]
            diff = abs(t-p)/t*100
            plist.append(diff)
        percent_diff_ave = sum(plist)/len(plist)
        mse_ave = mean_squared_error(targets, predictions)
        r_squared_ave = r2_score(targets, predictions)
        print("pearsonr_ave: ", pearsonr_ave)
        return percent_diff_ave, mse_ave, r_squared_ave
