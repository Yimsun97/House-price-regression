# import necessary packages
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import category_encoders as ce

from calculation_metrics import cal_pearsonr


def load_data(input_path):
    # read the input file
    df = pd.read_csv(input_path, header=0, index_col=0)
    return df


def save_data(test_y, cs_y):
    test_y = test_y.reshape(-1, 1)
    test_y_inv = cs_y.inverse_transform(test_y)
    test_y_inv = np.exp(test_y_inv)
    id = np.arange(1461, 2920)
    output = pd.DataFrame({'Id': id, 'SalePrice': test_y_inv.ravel()})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")


def preprocess_data(train, val, test):
    # feature definition
    features = train.columns.tolist()
    continuous_features = train.dtypes[train.dtypes.isin([np.dtype('int64'),
                                                          np.dtype('float64')])].index.tolist()
    continuous_features.remove('SalePrice')
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

    # extract target
    train_target = train.loc[:, target].copy()
    val_target = val.loc[:, target].copy()

    # continuous features
    # define imputer
    imputer = SimpleImputer(strategy='mean')
    train_continuous = imputer.fit_transform(train.loc[:, continuous_features])
    val_continuous = imputer.transform(val.loc[:, continuous_features])
    test_continuous = imputer.transform(test.loc[:, continuous_features])

    # # correlation filter
    # # rebuild index and columns
    # train_continuous = pd.DataFrame(train_continuous, columns=continuous_features, index=train.index)
    # val_continuous = pd.DataFrame(val_continuous, columns=continuous_features, index=val.index)
    # test_continuous = pd.DataFrame(test_continuous, columns=continuous_features, index=test.index)
    # pear_name = correlation_filter(train_continuous, train_target, filter_point=0.5, verbose=0)
    # train_continuous = train_continuous.loc[:, pear_name].values
    # val_continuous = val_continuous.loc[:, pear_name].values
    # test_continuous = test_continuous.loc[:, pear_name].values
    #
    # # perform encodings on continuous features
    # continuous_features_pseudo, enc = categorical_encodings(
    #     train_continuous, filter_point=5, method='catboost', train_target=train_target)
    # train_ctn = enc.transform(train_continuous[:, continuous_features_pseudo])
    # val_ctn = enc.transform(val_continuous[:, continuous_features_pseudo])
    # test_ctn = enc.transform(test_continuous[:, continuous_features_pseudo])
    # if len(train_ctn) > 0:
    #     train_continuous = np.hstack([train_continuous, train_ctn])
    #     val_continuous = np.hstack([val_continuous, val_ctn])
    #     test_continuous = np.hstack([test_continuous, test_ctn])

    # performing min-max scaling on each continuous feature column to the range [0, 1]
    cs_x = MinMaxScaler()
    train_continuous = cs_x.fit_transform(train_continuous)
    val_continuous = cs_x.transform(val_continuous)
    test_continuous = cs_x.transform(test_continuous)

    # perform log transformation and min-max scaling
    cs_y = MinMaxScaler()
    train_target = cs_y.fit_transform(np.log(train_target))
    val_target = cs_y.transform(np.log(val_target))

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

    # detect the outliers using IsolationTree
    outliers_fraction = 0.005
    train_data, train_target = iso_tree(train_data, train_target, outliers_fraction=outliers_fraction)
    val_data, val_target = iso_tree(val_data, val_target, outliers_fraction=outliers_fraction)

    # return the concatenated training and testing data
    return train_data, val_data, test_data, \
           train_target, val_target, cs_x, cs_y


def iso_tree(x, y, outliers_fraction=0.01):
    data = np.hstack([x, y])
    clf = IsolationForest(max_samples=data.shape[0], random_state=1,
                          contamination=outliers_fraction)
    preds = clf.fit_predict(data)
    x = x[preds == 1]
    y = y[preds == 1]
    return x, y


def correlation_filter(train_continuous, train_target, filter_point=0.5, verbose=0):
    # get continuous features
    continuous_features = train_continuous.columns.tolist()

    # correlation filter: only the variables whose abs(r) > filter_point are kept
    pear_name = []  # id
    pear_num = []  # r

    for i in continuous_features:
        r = cal_pearsonr(train_continuous.loc[:, i].values, train_target.values)
        if np.abs(r) > filter_point:
            pear_name.append(i)
            pear_num.append(r)
            if verbose:
                print("%s: %.4f" % (i, r))

    return pear_name


def categorical_encodings(train_continuous, filter_point=5, method='catboost', train_target=None):
    if method in ['catboost', 'target'] and train_target is None:
        raise ValueError('this method need train target')

    # screening the features whose items are less than filter_point
    continuous_features_pseudo = []
    for i in range(train_continuous.shape[1]):
        if len(set(train_continuous[:, i])) < filter_point:
            continuous_features_pseudo.append(i)

    # perform count encoding on continuous features whose items are less than 5
    train_ctn = []
    val_ctn = []
    test_ctn = []
    if len(continuous_features_pseudo) > 0:
        # count encoding
        if method == 'count':
            enc = ce.CountEncoder()
            enc.fit(train_continuous[:, continuous_features_pseudo])
        # catboost encoding
        elif method == 'catboost':
            enc = ce.CatBoostEncoder()
            enc.fit(train_continuous[:, continuous_features_pseudo], train_target)
        # target encoding
        elif method == 'target':
            enc = ce.TargetEncoder()
            enc.fit(train_continuous[:, continuous_features_pseudo], train_target)
        else:
            raise ValueError('method not supported')

    return continuous_features_pseudo, enc
