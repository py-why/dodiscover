from typing import Optional, Tuple

import networkx as nx
import pandas as pd

from dodiscover._protocol import EquivalenceClass
from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.constraint import SkeletonMethods
from dodiscover.context import Context
from dodiscover.typing import SeparatingSet

from .fcialg import FCI


class PsiFCI(FCI):
    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        alpha: float = 0.05,
        min_cond_set_size: Optional[int] = None,
        max_cond_set_size: Optional[int] = None,
        max_combinations: Optional[int] = None,
        skeleton_method: SkeletonMethods = SkeletonMethods.NBRS,
        apply_orientations: bool = True,
        max_iter: int = 1000,
        max_path_length: Optional[int] = None,
        pds_skeleton_method: SkeletonMethods = SkeletonMethods.PDS,
        known_intervention_targets: bool = False,
        **ci_estimator_kwargs,
    ):
        """Interventional (Psi) FCI algorithm.

        The I-FCI (or Psi-FCI) algorithm is an algorithm that accepts
        multiple sets of data that may pertain to observational and/or
        multiple interventional datasets under a known (I-FCI), or unknown (Psi-FCI)
        intervention target setting. Our API consolidates them here under
        one class, but you can control the setting using our hyperparameter.
        See :footcite:`Kocaoglu2019characterization` for more information on
        I-FCI and :footcite:`Jaber2020causal` for more information on Psi-FCI.

        The Psi-FCI algorithm is complete for the Psi-PAG equivalence class.
        However, the I-FCI has not been shown to be complete for the I-PAG
        equivalence class. Note that the I-FCI algorithm may change without
        notice.

        Parameters
        ----------
        ci_estimator : Callable
            The conditional independence test function. The arguments of the estimator should
            be data, node, node to compare, conditioning set of nodes, and any additional
            keyword arguments.
        alpha : float, optional
            The significance level for the conditional independence test, by default 0.05.
        min_cond_set_size : int, optional
            Minimum size of the conditioning set, by default None, which will be set to '0'.
            Used to constrain the computation spent on the algorithm.
        max_cond_set_size : int, optional
            Maximum size of the conditioning set, by default None. Used to limit
            the computation spent on the algorithm.
        max_combinations : int, optional
            The maximum number of conditional independence tests to run from the set
            of possible conditioning sets. By default None, which means the algorithm will
            check all possible conditioning sets. If ``max_combinations=n`` is set, then
            for every conditioning set size, 'p', there will be at most 'n' CI tests run
            before the conditioning set size 'p' is incremented. For controlling the size
            of 'p', see ``min_cond_set_size`` and ``max_cond_set_size``. This can be used
            in conjunction with ``keep_sorted`` parameter to only test the "strongest"
            dependences.
        skeleton_method : SkeletonMethods
            The method to use for testing conditional independence. Must be one of
            ('neighbors', 'complete', 'neighbors_path'). See Notes for more details.
        apply_orientations : bool
            Whether or not to apply orientation rules given the learned skeleton graph
            and separating set per pair of variables. If ``True`` (default), will
            apply Zhang's orientation rules R0-10, orienting colliders and certain
            arrowheads and tails :footcite:`Zhang2008`.
        max_iter : int
            The maximum number of iterations through the graph to apply
            orientation rules.
        max_path_length : int, optional
            The maximum length of any discriminating path, or None if unlimited.
        selection_bias : bool
            Whether or not to account for selection bias within the causal PAG.
            See :footcite:`Zhang2008`. Currently not implemented.
        pds_skeleton_method : SkeletonMethods
            The method to use for learning the skeleton using PDS. Must be one of
            ('pds', 'pds_path'). See Notes for more details.
        known_intervention_targets : bool, optional
            If `True`, then will run the I-FCI algorithm. If `False`, will run the
            Psi-FCI algorithm. By default False.
        ci_estimator_kwargs : dict
            Keyword arguments for the ``ci_estimator`` function.
        """
        super().__init__(
            ci_estimator,
            alpha,
            min_cond_set_size,
            max_cond_set_size,
            max_combinations,
            skeleton_method,
            apply_orientations,
            max_iter=max_iter,
            max_path_length=max_path_length,
            selection_bias=False,
            pds_skeleton_method=pds_skeleton_method,
            **ci_estimator_kwargs,
        )
        self.known_intervention_targets = known_intervention_targets

    def learn_skeleton(
        self, data: pd.DataFrame, context: Context, sep_set: Optional[SeparatingSet] = None
    ) -> Tuple[nx.Graph, SeparatingSet]:
        return super().learn_skeleton(data, context, sep_set)

    def fit(self, data: pd.DataFrame, context: Context) -> None:
        return super().fit(data, context)

    def orient_edges(self, graph: EquivalenceClass):
        return super().orient_edges(graph)

    def convert_skeleton_graph(self, graph: nx.Graph) -> EquivalenceClass:
        import pywhy_graphs as pgraph

        # convert the undirected skeleton graph to its PAG-class, where
        # all left-over edges have a "circle" endpoint
        if self.known_intervention_targets:
            pag = pgraph.IPAG(incoming_circle_edges=graph, name="IPAG derived with I-FCI")
        else:
            pag = pgraph.PsiPAG(incoming_circle_edges=graph, name="PsiPAG derived with Psi-FCI")
        return pag
