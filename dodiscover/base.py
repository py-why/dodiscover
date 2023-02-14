import inspect
import warnings
from collections import defaultdict
from copy import deepcopy

from ._version import __version__


class InconsistentVersionWarning(UserWarning):
    """Warning raised when an estimator is unpickled with a inconsistent version.

    Parameters
    ----------
    estimator_name : str
        Estimator name.

    current_dodiscover_version : str
        Current dodiscover version.

    original_dodiscover_version : str
        Original dodiscover version.
    """

    def __init__(self, *, estimator_name, current_dodiscover_version, original_dodiscover_version):
        self.estimator_name = estimator_name
        self.current_dodiscover_version = current_dodiscover_version
        self.original_dodiscover_version = original_dodiscover_version

    def __str__(self):
        return (
            f"Trying to unpickle estimator {self.estimator_name} from version"
            f" {self.original_dodiscover_version} when "
            f"using version {self.current_dodiscover_version}. This might lead to breaking"
            " code or "
            "invalid results. Use at your own risk. "
        )


class BasePyWhy:
    """Base class for all PyWhy class objects.

    TODO: add parameter validation and data validation from sklearn.
    TODO: add HTML representation.

    Notes
    -----
    All learners and context should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "dodiscover estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this Context.

        TODO: can update this when we build a causal-Pipeline similar to sklearn's Pipeline.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)

            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                # future proof for pipeline objects
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            elif deep and not isinstance(value, type):
                # this ensures a deepcopy is applied, which is useful for graphs
                value = deepcopy(value)

            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : instance
            Learner instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __getstate__(self):
        if getattr(self, "__slots__", None):
            raise TypeError(
                "You cannot use `__slots__` in objects inheriting from "
                "`dodiscover.base.BasePyWhy`."
            )

        try:
            state = super().__getstate__()
            if state is None:
                # For Python 3.11+, empty instance (no `__slots__`,
                # and `__dict__`) will return a state equal to `None`.
                state = self.__dict__.copy()
        except AttributeError:
            # Python < 3.11
            state = self.__dict__.copy()

        if type(self).__module__.startswith("dodiscover."):
            return dict(state.items(), _dodiscover_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith("dodiscover."):
            pickle_version = state.pop("_dodiscover_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    InconsistentVersionWarning(
                        estimator_name=self.__class__.__name__,
                        current_dodiscover_version=__version__,
                        original_dodiscover_version=pickle_version,
                    ),
                )
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)
