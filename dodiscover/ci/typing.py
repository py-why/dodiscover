from typing import Protocol, Union

import sklearn


class PyTorchModel(Protocol):
    def forward(self):
        pass


Classifier = Union[sklearn.base.BaseEstimator, PyTorchModel]
