# import necessary packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import seaborn as sns; sns.set()
from xgboost import XGBRegressor

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from datasets import load_data, preprocess_data, cal_rms, save_data, model_evaluation
from pt_net import RegressionNet, net_train
from visualization import train_loss_vis, pred_test_vis

# set the seed
np.random.seed(seed=1)
torch.manual_seed(seed=1)

# specify data path
train_path = 'house-prices-advanced-regression-techniques/train.csv'
test_path = 'house-prices-advanced-regression-techniques/test.csv'
train = load_data(train_path)
test = load_data(test_path)

train_set, val_set = train_test_split(train, test_size=0.2, random_state=2)

# %%
# feature engineering here
print("[INFO] processing data...")
(train_x, val_x, test_x,
 train_y, val_y, cs_x, cs_y) = preprocess_data(train_set, val_set, test)

# build the structure of the net
num_in = train_x.shape[1]
num_h1 = 8
num_h2 = 8
num_h3 = 8
num_epoch = 200
num_batch = 5

net = RegressionNet(num_in, num_h1, num_h2, num_h3)
print(net)
# %%
# pick the optimizer and loss function
optimizer = torch.optim.Adam(net.parameters(), weight_decay=7e-4)
# optimizer = torch.optim.RMSprop(net.parameters())

loss_fun = torch.nn.MSELoss()

# convert numpy array to tensor
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
val_x = torch.from_numpy(val_x).float()
val_y = torch.from_numpy(val_y).float()

# train and record the loss
net, train_loss_list, val_loss_list = net_train(net, train_x, train_y, num_batch,
                                                num_epoch, loss_fun, optimizer,
                                                test_x=val_x, test_y=val_y)
# %%
# evaluate the model
train_pred = net(train_x)
val_pred = net(val_x)

# tensor to numpy
val_x = val_x.detach().numpy()
val_y = val_y.detach().numpy()
val_pred = val_pred.detach().numpy()
train_x = train_x.detach().numpy()
train_y = train_y.detach().numpy()
train_pred = train_pred.detach().numpy()

# calculate evaluation indices
print("[INFO]---NN----------")
model_evaluation(train_y, train_pred, val_y, val_pred, cs_y)
# print("[INFO] train loss: {:.6f}".format(train_loss_list[-1]))
# print("[INFO] validation loss: {:.6f}".format(val_loss_list[-1]))
# save_data(test_x, net.to_model, cs_y)

# visualization
# plt.close('all')
train_loss_vis(train_loss_list, val_loss_list)
pred_test_vis(val_pred, val_y)

#%%
# XGBoost
model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                         learning_rate=0.05, max_depth=3,
                         min_child_weight=1.7817, n_estimators=2200,
                         reg_alpha=0.4640, reg_lambda=0.8571,
                         subsample=0.5213, random_state=7, nthread=-1)
model_xgb.fit(train_x, train_y)
train_pred_xgb = model_xgb.predict(train_x).reshape(-1, 1)
val_pred_xgb = model_xgb.predict(val_x).reshape(-1, 1)
print("[INFO]---XGB---------")
model_evaluation(train_y=train_y, train_pred=train_pred_xgb, val_y=val_y,
                 val_pred=val_pred_xgb, cs_y=cs_y)
pred_test_vis(val_pred_xgb, val_y)

#%%
# model blending


def blending(x):
    nn_weight = 0.8
    xgb_weight = 1 - nn_weight
    return net.to_model(x) * nn_weight + model_xgb.predict(x).reshape(-1, 1) * xgb_weight


train_pred_blending = blending(train_x)
val_pred_blending = blending(val_x)
print("[INFO]---BLE---------")
model_evaluation(train_y, train_pred_blending, val_y, val_pred_blending, cs_y)
pred_test_vis(val_pred_blending, val_y)

save_data(test_x, blending, cs_y)
