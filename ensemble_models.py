# import necessary packages
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold


class AveragingModels(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models
        self.models_ = []

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.base_models_ = []
        self.meta_model_ = []
        self.scores = []

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.scores = [list() for x in self.base_models]
        # clone the base model
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # train the cloned base model and extract the new features
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            # split the data into training and test data
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])

                # record the scores on the validation set
                self.scores[i].append(instance.score(X[holdout_index], y[holdout_index]))
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # use meta model to train the results of base models
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def blending(x, models, model_weights=0.8):
    if isinstance(model_weights, float):
        model_weights = [model_weights]

    if len(models) == len(model_weights) + 1:
        model_weights.append(1 - sum(model_weights))
    elif len(models) == len(model_weights):
        if sum(model_weights) != 1:
            raise ValueError("sum of weights does not equal 1")
    else:
        raise ValueError("wrong length of model weights")

    model_weights = np.array(model_weights).reshape(-1, 1)

    blending_out = np.array([model.predict(x) for model in models]).T
    blending_out = np.matmul(blending_out, model_weights)

    return blending_out
