# import necessary packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data


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


class RegressionNet(nn.Module):

    def __init__(self, num_in, *args):
        super().__init__()
        self.hidden_list = list(args)
        for i, hidden in enumerate(self.hidden_list):
            if i == 0:
                self.__setattr__(f'hidden_{i}',
                                 nn.Linear(num_in, self.hidden_list[i]))
            elif i < len(self.hidden_list):
                self.__setattr__(f'hidden_{i}',
                                 nn.Linear(self.hidden_list[i - 1], self.hidden_list[i]))
            else:
                break

        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(self.hidden_list[-1], 1)

    def forward(self, x):
        for i, hidden in enumerate(self.hidden_list):
            if i >= len(self.hidden_list):
                break
            x = self.__getattr__(f'hidden_{i}')(x)     # Linear
            # x = self.dropout(x)                      # Dropout(0.3)
            x = F.relu(x)                              # ReLU

        x = self.output(x)
        return x

    def to_model(self, x):
        self.eval()
        return self.forward(torch.from_numpy(np.array(x)).float()).detach().numpy()


def net_train(net, train_x, train_y, num_batch, num_epoch, loss_fun, optimizer,
              test_x=None, test_y=None):
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
            net.train()
            batch_pred = net(batch_x)
            batch_loss = loss_fun(batch_pred, batch_y)
            batch_loss_accumulator += batch_loss.item() * batch_x.size(0)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss_list.append(batch_loss_accumulator / train_x.size(0))

        if not(test_x is None) and not(test_y is None):
            net.eval()
            val_pred = net(test_x)
            val_loss = loss_fun(val_pred, test_y)
            val_loss_list.append(val_loss.item())

    return net, train_loss_list, val_loss_list
