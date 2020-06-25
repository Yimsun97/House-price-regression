# import necessary packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from calculation_metrics import model_evaluation
from datasets import load_data, preprocess_data
from ensemble_models import blending
from visualization import pred_test_vis


class ModelWrapper(object):
    def __init__(self, model_name, model_object, train_x, train_y, cs_y,
                 val_x=None, val_y=None):
        self.model_name = model_name
        self.model_object = model_object
        self.train_x = train_x
        self.train_y = train_y
        self.cs_y = cs_y

        if val_x is not None:
            self.val_x = val_x
        if val_y is not None:
            self.val_y = val_y

    def fit(self, X=None, y=None):
        if X is None or y is None:
            X, y = self.train_x, self.train_y

        self.model_object.fit(X, y)
        return self

    def predict(self, X=None):
        if X is None:
            X = self.val_x
        return self.model_object.predict(X).reshape(-1, 1)

    def evaluate(self):
        train_pred = self.predict(self.train_x)
        val_pred = self.predict(self.val_x)
        cs_y = self.cs_y
        model_evaluation(self.train_y, train_pred, self.val_y,
                         val_pred, cs_y, self.model_name)

    def visualize(self):
        val_pred = self.predict(self.val_x)
        pred_test_vis(val_pred, self.val_y)


def normal_model(model, model_name, train_x, train_y, val_x, val_y, cs_y, model_weights=None):
    if isinstance(model, list):
        if model_weights is not None:
            # feed data in first
            for m in model:
                m.fit(train_x, train_y.ravel())

            train_pred = blending(train_x, model, model_weights)
            val_pred = blending(val_x, model, model_weights)
        else:
            raise ValueError('models should have weighs assigned.')
    else:
        model.fit(train_x, train_y.ravel())
        train_pred = model.predict(train_x).reshape(-1, 1)
        val_pred = model.predict(val_x).reshape(-1, 1)

    model_evaluation(train_y=train_y, train_pred=train_pred, val_y=val_y,
                     val_pred=val_pred, cs_y=cs_y, bnd=model_name)
    pred_test_vis(val_pred, val_y)

    return model


if __name__ == '__main__':
    # load data
    train_path = 'house-prices-advanced-regression-techniques/train.csv'
    train_data = load_data(train_path)

    # split training and validation set
    train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=2)

    # feature engineering
    print("[INFO] processing data...")
    (train_x, val_x, test_x,
     train_y, val_y, cs_x, cs_y) = preprocess_data(train_set, val_set, train_set)

    xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                       learning_rate=0.05, max_depth=3,
                       min_child_weight=1.7817, n_estimators=2200,
                       reg_alpha=0.4640, reg_lambda=0.8571,
                       subsample=0.5213, random_state=7, nthread=-1)

    normal_model(xgb, 'xgb', train_x, train_y, val_x, val_y, cs_y)

