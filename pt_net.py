# import necessary packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.base import BaseEstimator, RegressorMixin


class OldRegressionNet(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3):
        super().__init__()
        self.hidden_0 = nn.Linear(num_in, num_h1)
        self.hidden_1 = nn.Linear(num_h1, num_h2)
        self.hidden_2 = nn.Linear(num_h2, num_h3)
        self.output = nn.Linear(num_h3, 1)

    def forward(self, x_in):
        x_out = F.relu(self.hidden_1(x_in))
        x_out = F.relu(self.hidden_2(x_out))
        x_out = F.relu(self.hidden_3(x_out))
        x_out = self.output(x_out)
        return x_out


class RegressionNet(nn.Module, BaseEstimator, RegressorMixin):

    def __init__(self, num_in, *args):
        super().__init__()
        self.num_in = num_in
        self.args = args
        self.hidden_list = list(self.args)
        for i, hidden in enumerate(self.hidden_list):
            if i == 0:
                self.__setattr__(f'hidden_{i}',
                                 nn.Linear(self.num_in, self.hidden_list[i]))
            elif i < len(self.hidden_list):
                self.__setattr__(f'hidden_{i}',
                                 nn.Linear(self.hidden_list[i - 1], self.hidden_list[i]))
            else:
                break

        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(self.hidden_list[-1], 1)
        self.train_loss_list = []
        self.val_loss_list = []

    def forward(self, x):
        for i, hidden in enumerate(self.hidden_list):
            if i >= len(self.hidden_list):
                break
            x = self.__getattr__(f'hidden_{i}')(x)     # Linear
            # x = self.dropout(x)                      # Dropout(0.3)
            x = F.relu(x)                              # ReLU

        x = self.output(x)
        return x

    def fit(self, train_x, train_y, num_batch=5, num_epoch=200, weight_decay=7e-4,
            loss_fun=None, optimizer=None, test_x=None, test_y=None):
        if loss_fun is None:
            loss_fun = torch.nn.MSELoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay)

        if len(train_y.shape) == 1:
            train_y = train_y.reshape(-1, 1)

        if isinstance(train_x, np.ndarray):
            train_x = torch.from_numpy(train_x).float()
        if isinstance(train_y, np.ndarray):
            train_y = torch.from_numpy(train_y).float()

        # generate the data loader
        torch_dataset = Data.TensorDataset(train_x, train_y)
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=num_batch,  # mini batch size
            shuffle=True,  # shuffle the data or not (shuffle=True is better)
        )

        # record the loss
        train_loss_list = []
        val_loss_list = []

        # train the model and calculate train loss and validation loss
        for epoch in range(num_epoch):
            batch_loss_accumulator = 0.0
            for batch, (batch_x, batch_y) in enumerate(loader):
                self.train()
                batch_pred = self.forward(batch_x)
                batch_loss = loss_fun(batch_pred, batch_y)
                batch_loss_accumulator += batch_loss.item() * batch_x.size(0)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            train_loss_list.append(batch_loss_accumulator / train_x.size(0))

            if (test_x is not None) and (test_y is not None):
                self.eval()
                if isinstance(test_x, np.ndarray):
                    test_x = torch.from_numpy(test_x).float()
                if isinstance(test_y, np.ndarray):
                    test_y = torch.from_numpy(test_y).float()

                val_pred = self.forward(test_x)
                val_loss = loss_fun(val_pred, test_y)
                val_loss_list.append(val_loss.item())

        self.train_loss_list += train_loss_list
        if (test_x is not None) and (test_y is not None):
            self.val_loss_list += val_loss_list

        return self

    def predict(self, data):
        self.eval()
        return self.forward(torch.from_numpy(np.array(data)).float())\
            .detach().numpy().ravel()


class SKRegressionNet(RegressionNet):

    def __init__(self, num_in, num_h1, num_h2, num_h3):
        super().__init__(num_in, num_h1, num_h2, num_h3)
        self.num_in = num_in
        self.num_h1 = num_h1
        self.num_h2 = num_h2
        self.num_h3 = num_h3

    def get_params(self, deep=True):
        return {'num_in': self.num_in, 'num_h1': self.num_h1,
                'num_h2': self.num_h2, 'num_h3': self.num_h3}

