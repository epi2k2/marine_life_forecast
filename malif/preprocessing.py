from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
import pandas as pd


class MultilabelEncoder(BaseEstimator, TransformerMixin):
    """Target encoding for categorical features for multi label targets.

    For the case of categorical target: features are replaced with 
    a blend of posterior probability of the target given particular categorical 
    value and the prior probability of the target over all the training data.

    For the case of continuous target: features are replaced with 
    a blend of the expected value of the target given particular categorical 
    value and the expected value of the target over all the training data.
    """

    def __init__(self, encoders=[]):

        super().__init__()
        self.encoders = encoders

    def fit(self, X, Y):
        """Fit encoder according to X and Y
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        Y : array-like, shape = [n_samples, n_targets]
            Target values.
        Returns
        -------
        self: encoder
            Returns self.
        """

        self.encoders = []

        for n, y in enumerate(Y):
            self.encoders.append(TargetEncoder())
            self.encoders[n].fit(X, Y[y])

        return self

    def transform(self, X):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vectors to transform, where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        p: array, shape = [n_samples, n_numeric + N] 
            Transformed values with encoding applied.
        """

        outputs = []

        for n in range(len(self.encoders)):
            outputs.append(self.encoders[n].transform(X))

        for n in range(len(outputs)):
            columns = dict([(column, column+'_'+str(n))
                           for column in outputs[n].columns])
            outputs[n].rename(columns=columns, inplace=True)

            if n == 0:
                output = outputs[0].copy()
            else:
                output = pd.concat([output, outputs[n]], axis=1)

        return output
