# import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from datasets import load_data, preprocess_data, save_data
from model_candidates import normal_model
from pt_net import RegressionNet, SKRegressionNet
from ensemble_models import AveragingModels, StackingAveragedModels, blending
from calculation_metrics import cal_rms, model_evaluation
from visualization import train_loss_vis, pred_test_vis

# prerequisites
# set the seed
np.random.seed(seed=1)
torch.manual_seed(seed=1)

# specify data path
train_path = 'house-prices-advanced-regression-techniques/train.csv'
test_path = 'house-prices-advanced-regression-techniques/test.csv'
train = load_data(train_path)
test = load_data(test_path)

# split training and validation set
train_set, val_set = train_test_split(train, test_size=0.2, random_state=2)

# %% feature engineering
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

# %% train nn
# pick the optimizer and loss function
weight_decay = 7e-4
optimizer = torch.optim.Adam(net.parameters(), weight_decay=weight_decay)
# optimizer = torch.optim.RMSprop(net.parameters())

loss_fun = torch.nn.MSELoss()

# convert numpy array to tensor
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
val_x = torch.from_numpy(val_x).float()
val_y = torch.from_numpy(val_y).float()

# train and record the loss
net.fit(train_x, train_y, num_batch=num_batch, num_epoch=num_epoch,
        loss_fun=loss_fun, optimizer=optimizer, test_x=val_x, test_y=val_y)

train_loss_list = net.train_loss_list
val_loss_list = net.val_loss_list

# %% evaluate the model
train_pred = net.predict(train_x).reshape(-1, 1)
val_pred = net.predict(val_x).reshape(-1, 1)

# tensor to numpy
val_x = val_x.detach().numpy()
val_y = val_y.detach().numpy()
train_x = train_x.detach().numpy()
train_y = train_y.detach().numpy()

# calculate evaluation indices
model_evaluation(train_y, train_pred, val_y, val_pred, cs_y, 'nn')

# save_data(net.predict(test_x), cs_y)

# visualization
train_loss_vis(train_loss_list, val_loss_list)
pred_test_vis(val_pred, val_y)

# %% XGBoost
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                   learning_rate=0.05, max_depth=3,
                   min_child_weight=1.7817, n_estimators=2200,
                   reg_alpha=0.4640, reg_lambda=0.8571,
                   subsample=0.5213, random_state=7, nthread=-1)
normal_model(xgb, 'xgb', train_x, train_y, val_x, val_y, cs_y)
# %% Random Forest
rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                           max_features=240, random_state=1)
normal_model(rf, 'rf', train_x, train_y, val_x, val_y, cs_y)
# %% model blending
normal_model([net, xgb, rf], 'net+xgb+rf', train_x, train_y,
             val_x, val_y, cs_y, [0.7, 0.2])

# %% stacking using split results
net_cv = SKRegressionNet(num_in, num_h1, num_h2, num_h3)
base_models = (net_cv, xgb, rf)
meta_model = LinearRegression()

avg_model = AveragingModels(base_models)
stacking_model = StackingAveragedModels(base_models, meta_model)

normal_model(stacking_model, 'stk', train_x, train_y, val_x, val_y, cs_y)

# save the data
save_data(stacking_model.predict(test_x), cs_y)
