from typing import Optional
import pandas as pd


def _forward_step(data, context, max_parents, metric):
    pass

def _backward_step(data, context, max_parents, metric):
    pass

def _turning_step(data, context, max_parents, metric):
    pass


class GreedyEquivalentSearch():
    def __init__(self,
                 forward_phase: bool = True,
                 backward_phase: bool = True,
                 turning_phase: bool = True,
                 max_parents: Optional[int] = None,
                 metric='bic') -> None:
        """_summary_

        Implements the GES algorithm initially introduced in :footcite:`chickering2002optimal`.

        Parameters
        ----------
        forward_phase : bool, optional
            _description_, by default True
        backward_phase : bool, optional
            _description_, by default True
        turning_phase : bool, optional
            _description_, by default True
        max_parents : Optional[int], optional
            _description_, by default None
        metric : str, optional
            _description_, by default 'bic'

        Notes
        -----
        The turning step was introduced in :footcite:`hauser2012characterization`.

        Other improvements were made in improving the runtime efficiency of the search algorithm,
        such as :footcite:`chickering2015selective` and :footcite:`chickering2020statistically`.

        References
        ----------
        .. footbibliography::
        """
        self.forward_phase = forward_phase
        self.backward_phase = backward_phase
        self.turning_phase = turning_phase
        self.max_parents = max_parents
        self.metric = metric

    def learn_graph(self, data: pd.DataFrame, context):
        pass