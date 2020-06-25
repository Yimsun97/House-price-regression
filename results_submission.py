# import necessary packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from datasets import save_data, preprocess_data, load_data
from ensemble_models import StackingAveragedModels
from pt_net import SKRegressionNet

# specify data path
train_path = 'house-prices-advanced-regression-techniques/train.csv'
test_path = 'house-prices-advanced-regression-techniques/test.csv'
train = load_data(train_path)
test = load_data(test_path)

# deal with features
(train_x_all, val_x_all, test_x_all,
 train_y_all, val_y_all, cs_x_all, cs_y_all) = preprocess_data(train, train, test)

# stacking using all the training data
num_h1 = 8
num_h2 = 8
num_h3 = 8
net_all = SKRegressionNet(train_x_all.shape[1], num_h1, num_h2, num_h3)
xgb_all = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                       learning_rate=0.05, max_depth=3,
                       min_child_weight=1.7817, n_estimators=2200,
                       reg_alpha=0.4640, reg_lambda=0.8571,
                       subsample=0.5213, random_state=7, nthread=-1)
rf_all = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                               max_features=240, random_state=1)

base_models = (net_all, xgb_all, rf_all)
meta_model = LinearRegression()
stacking_model = StackingAveragedModels(base_models, meta_model, n_folds=10)
stacking_model.fit(train_x_all, train_y_all.ravel())

# save the data
save_data(stacking_model.predict(test_x_all), cs_y_all)
