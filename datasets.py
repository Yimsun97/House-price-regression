from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import category_encoders as ce


def load_data(input_path):
    # read the input file
    df = pd.read_csv(input_path, header=0, index_col=0)
    return df


def preprocess_data(train, val, test=None):
    # feature definition
    features = train.columns.tolist()
    continuous_features = ["MSSubClass", "LotFrontage", "LotArea",
                           "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea",
                           "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                           "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
                           "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
                           "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea",
                           "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch",
                           "PoolArea", "MiscVal", "MoSold", "YrSold"]
    remove_features = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
    target = ["SalePrice"]
    categorical_features = [x for x in features if x not in continuous_features
                            + remove_features + target]
    # removed features
    train = train.drop(columns=remove_features).copy()
    val = val.drop(columns=remove_features).copy()
    test = test.drop(columns=remove_features).copy()

    # fill na
    train.fillna(method='ffill', inplace=True)
    val.fillna(method='ffill', inplace=True)
    test.fillna(method='ffill', inplace=True)

    # target: perform log transformation and min-max scaling
    cs_y = MinMaxScaler()
    train_target = cs_y.fit_transform(np.log(train.loc[:, target]))
    val_target = cs_y.transform(np.log(val.loc[:, target]))

    # continuous features
    # define imputer
    imputer = SimpleImputer(strategy='mean')
    train_continuous = imputer.fit_transform(train.loc[:, continuous_features])
    val_continuous = imputer.transform(val.loc[:, continuous_features])
    test_continuous = imputer.transform(test.loc[:, continuous_features])

    # # correlation filter: only the variables whose abs(r) > 0.5 are kept
    # pear_name = []  # id
    # pear_num = []  # r
    # train_continuous = pd.DataFrame(train_continuous, columns=continuous_features, index=train.index)
    # val_continuous = pd.DataFrame(val_continuous, columns=continuous_features, index=val.index)
    # test_continuous = pd.DataFrame(test_continuous, columns=continuous_features, index=test.index)
    #
    # for i in continuous_features:
    #     r = cal_pearsonr(train_continuous.loc[:, i], train_target)
    #     if np.abs(r) > 0.5:
    #         pear_name.append(i)
    #         pear_num.append(r)
    #         print("%s: %.4f" % (i, r))
    # train_continuous = train_continuous.loc[:, pear_name].values
    # val_continuous = val_continuous.loc[:, pear_name].values
    # test_continuous = test_continuous.loc[:, pear_name].values

    # # screening the features whose items are less than 5
    # continuous_features_1 = []
    # for i in range(train_continuous.shape[1]):
    #     if len(set(train_continuous[:, i])) < 5:
    #         continuous_features_1.append(i)
    #
    # # perform count encoding on continuous features whose items are less than 5
    # train_ctn = []
    # val_ctn = []
    # test_ctn = []
    # if len(continuous_features_1) > 0:
    #     # count encoding
    #     # count_enc = ce.CountEncoder()
    #     # count_enc.fit(train_continuous[:, continuous_features_1])
    #     # train_ctn = count_enc.transform(train_continuous[:, continuous_features_1])
    #     # val_ctn = count_enc.transform(val_continuous[:, continuous_features_1])
    #     # test_ctn = count_enc.transform(test_continuous[:, continuous_features_1])
    #
    #     # catboost encoding
    #     cat_enc = ce.CatBoostEncoder()
    #     cat_enc.fit(train_continuous[:, continuous_features_1], train.loc[:, target])
    #     train_ctn = cat_enc.transform(train_continuous[:, continuous_features_1])
    #     val_ctn = cat_enc.transform(val_continuous[:, continuous_features_1])
    #     test_ctn = cat_enc.transform(test_continuous[:, continuous_features_1])
    #
    # if len(train_ctn) > 0:
    #     train_continuous = np.hstack([train_continuous, train_ctn])
    #     val_continuous = np.hstack([val_continuous, val_ctn])
    #     test_continuous = np.hstack([test_continuous, test_ctn])

    # performing min-max scaling on each continuous feature column to the range [0, 1]
    cs_x = MinMaxScaler()
    train_continuous = cs_x.fit_transform(train_continuous)
    val_continuous = cs_x.transform(val_continuous)
    test_continuous = cs_x.transform(test_continuous)

    # categorical features
    # performing label binarizer in each categorical feature column
    train_categorical = []
    val_categorical = []
    test_categorical = []
    for i, item in enumerate(categorical_features):
        ctg = LabelBinarizer()
        train_ctg = ctg.fit_transform(train.loc[:, item])
        val_ctg = ctg.transform(val.loc[:, item])
        test_ctg = ctg.transform(test.loc[:, item])
        if i == 0:
            train_categorical = train_ctg
            val_categorical = val_ctg
            test_categorical = test_ctg
        else:
            train_categorical = np.hstack([train_categorical, train_ctg])
            val_categorical = np.hstack([val_categorical, val_ctg])
            test_categorical = np.hstack([test_categorical, test_ctg])

    # construct our training and testing data
    if len(train_categorical) > 0:
        train_data = np.hstack([train_continuous, train_categorical])
        val_data = np.hstack([val_continuous, val_categorical])
        test_data = np.hstack([test_continuous, test_categorical])
    else:
        train_data = train_continuous
        val_data = val_continuous
        test_data = test_continuous

    # detest the outliers using IsolationTree
    train_data, train_target = iso_tree(train_data, train_target, outliers_fraction=0.005)
    val_data, val_target = iso_tree(val_data, val_target, outliers_fraction=0.005)

    # return the concatenated training and testing data
    return train_data, val_data, test_data, \
           train_target, val_target, cs_x, cs_y


def cal_rms(y_true, y_pred, cs_y):
    y_true_inv = cs_y.inverse_transform(y_true)
    y_pred_inv = cs_y.inverse_transform(y_pred)
    y_true_inv = np.exp(y_true_inv)
    y_pred_inv = np.exp(y_pred_inv)
    mse = mean_squared_error(np.log(y_true_inv), np.log(y_pred_inv))
    return np.sqrt(mse)


def save_data(test_x, model, cs_y):
    test_y = model(test_x).reshape(-1, 1)
    test_y_inv = cs_y.inverse_transform(test_y)
    test_y_inv = np.exp(test_y_inv)
    id = np.arange(1461, 2920)
    output = pd.DataFrame({'Id': id, 'SalePrice': test_y_inv.ravel()})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")


def cal_pearsonr(x, y):
    if len(x.shape) > 1:
        x = x.ravel()
    elif len(y.shape) > 1:
        y = y.ravel()

    r, _ = pearsonr(x, y)
    return r


def iso_tree(x, y, outliers_fraction=0.01):
    data = np.hstack([x, y])
    clf = IsolationForest(max_samples=data.shape[0], random_state=1,
                          contamination=outliers_fraction)
    preds = clf.fit_predict(data)
    x = x[preds == 1]
    y = y[preds == 1]
    return x, y


def model_evaluation(train_y, train_pred, val_y, val_pred, cs_y):
    r2_train = r2_score(train_y, train_pred)
    r2_val = r2_score(val_y, val_pred)
    print("[INFO] r2_score of train: {:.4f}".format(r2_train))
    print("[INFO] r2_score of validation: {:.4f}".format(r2_val))
    print("[INFO] training RMSE: {:.6f}".format(cal_rms(train_y, train_pred, cs_y)))
    print("[INFO] validation RMSE: {:.6f}".format(cal_rms(val_y, val_pred, cs_y)))
    pass
