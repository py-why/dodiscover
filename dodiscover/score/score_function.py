from typing import Callable

import pandas as pd


class ScoreFunction:
    def __init__(self, score: Callable) -> None:
        self._cache = {}
        self.score_func = score

    def local_score(self, data: pd.DataFrame, source, source_parents) -> float:
        """Compute the local score of an edge.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset.
        source : Node
            The origin node.
        source_parents : list of Node
            The parents of the source.

        Returns
        -------
        float
            The score.
        """
        # key is a tuple of the form (source, sorted(source_parents))
        key = (source, tuple(sorted(source_parents)))

        try:
            score = self._cache[key]
        except KeyError:
            score = self.score_func(data, source, source_parents)
            self._cache[key] = score
        return score
