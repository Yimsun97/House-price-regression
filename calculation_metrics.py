# import necessary packages
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def cal_rms(y_true, y_pred, cs_y):
    y_true_inv = cs_y.inverse_transform(y_true)
    y_pred_inv = cs_y.inverse_transform(y_pred)
    y_true_inv = np.exp(y_true_inv)
    y_pred_inv = np.exp(y_pred_inv)
    mse = mean_squared_error(np.log(y_true_inv), np.log(y_pred_inv))
    return np.sqrt(mse)


def cal_pearsonr(x, y):
    if len(x.shape) > 1:
        x = x.ravel()
    elif len(y.shape) > 1:
        y = y.ravel()

    r, _ = pearsonr(x, y)
    return r


def model_evaluation(train_y, train_pred, val_y, val_pred, cs_y, bnd):
    r2_train = r2_score(train_y, train_pred)
    r2_val = r2_score(val_y, val_pred)
    print("[INFO]---{:s}---------".format(bnd.upper()))
    print("[INFO] r2_score of train: {:.4f}".format(r2_train))
    print("[INFO] r2_score of validation: {:.4f}".format(r2_val))
    print("[INFO] training RMSE: {:.6f}".format(cal_rms(train_y, train_pred, cs_y)))
    print("[INFO] validation RMSE: {:.6f}".format(cal_rms(val_y, val_pred, cs_y)))

