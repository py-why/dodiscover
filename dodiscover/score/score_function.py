from typing import Callable, Dict

import numpy as np
import pandas as pd


class ScoreFunction:
    def __init__(self, score: Callable) -> None:
        self._cache: Dict = dict()
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

    # XXX: this is only for Likelihood score for Gaussian data
    def full_score(self, A):
        """Full score of a DAG.

        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a sample from a single environment, by finding the maximum
        likelihood estimates of the corresponding connectivity matrix
        (weights) and noise term variances.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.

        Returns
        -------
        score : float
            the penalized log-likelihood score.

        """
        # Compute MLE
        B, omegas = self._mle_full(A)
        # Compute log-likelihood (without log(2π) term)
        K = np.diag(1 / omegas)
        I_B = np.eye(self.p) - B.T
        log_term = self.n * np.log(omegas.prod())
        if self.method == "scatter":
            # likelihood = 0.5 * self.n * (np.log(det_K) - np.trace(K @ I_B @ self._scatter @ I_B.T))
            likelihood = log_term + self.n * np.trace(K @ I_B @ self._scatter @ I_B.T)
        else:
            # Center the data, exclude the intercept column
            inv_cov = I_B.T @ K @ I_B
            cov_term = 0
            for i, x in enumerate(self._centered):
                cov_term += x @ inv_cov @ x
            likelihood = log_term + cov_term
        #   Note: the number of parameters is the number of edges + the p marginal variances
        l0_term = self.lmbda * (np.sum(A != 0) + 1 * self.p)
        score = -0.5 * likelihood - l0_term
        return score
