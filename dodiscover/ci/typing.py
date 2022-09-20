from typing import Protocol, Union

import sklearn


class SkorchModel(Protocol):
    """PyTorch model compliant with scikit-learn API.

    PyTorch models are not sklearn-compliant out of the box,
    but a skorch model is a light-weight wrapper that transforms
    any PyTorch model into a Neural Network model compliant with
    the sklearn API.
    """

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pass


Classifier = Union[sklearn.base.BaseEstimator, SkorchModel]
