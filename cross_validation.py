# import necessary packages
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from calculation_metrics import cal_rms
from datasets import load_data, preprocess_data
from ensemble_models import blending
from model_candidates import normal_model
from pt_net import SKRegressionNet

train_path = 'house-prices-advanced-regression-techniques/train.csv'
train = load_data(train_path)

# split training and validation set
train_set, val_set = train_test_split(train, test_size=0.2, random_state=2)

num_in = 265
num_h1 = 8
num_h2 = 8
num_h3 = 8
# 5 fold CV to test blending model
scores_cv = []
rmse_cv = []
net_cv = SKRegressionNet(num_in, num_h1, num_h2, num_h3)
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                   learning_rate=0.05, max_depth=3,
                   min_child_weight=1.7817, n_estimators=2200,
                   reg_alpha=0.4640, reg_lambda=0.8571,
                   subsample=0.5213, random_state=7, nthread=-1)

cv = KFold(n_splits=5, random_state=2, shuffle=True)

for train_index, val_index in cv.split(train):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", val_index)
    train_set_cv = train.iloc[train_index]
    val_set_cv = train.iloc[val_index]

    (train_x_cv, val_x_cv, test_x_cv,
     train_y_cv, val_y_cv,
     cs_x_cv, cs_y_cv) = preprocess_data(train_set, val_set, train_set)

    net_cv, xgb = normal_model([net_cv, xgb], 'net_cv+xgb', train_x_cv, train_y_cv,
                               val_x_cv, val_y_cv, cs_y_cv, [0.8, 0.2])

    val_pred_cv = blending(val_x_cv, [net_cv, xgb])
    scores_cv.append(r2_score(val_y_cv, val_pred_cv))
    rmse_cv.append(cal_rms(val_y_cv, val_pred_cv, cs_y_cv))

print("[INFO]---CV----------")
print("CV r2_score: {:0.2f} (+/- {:0.2f})".format(np.mean(scores_cv),
                                                  np.std(scores_cv) * 2))
print("CV RMSE: {:0.4f} (+/- {:0.4f})".format(np.mean(rmse_cv),
                                              np.std(rmse_cv) * 2))
