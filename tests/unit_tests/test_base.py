import pytest

from dodiscover.base import BasePyWhy

# TODO: add pickling tests


class MyLearner(BasePyWhy):
    def __init__(self, l1=0, empty=None):
        self.l1 = l1
        self.empty = empty


class K(BasePyWhy):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d


class T(BasePyWhy):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


def test_str():
    # Smoke test the str of the base estimator
    my_estimator = MyLearner()
    str(my_estimator)


def test_get_params():
    test = T(K(), K)

    assert "a__d" in test.get_params(deep=True)
    assert "a__d" not in test.get_params(deep=False)

    test.set_params(a__d=2)
    assert test.a.d == 2

    with pytest.raises(ValueError):
        test.set_params(a__a=2)
