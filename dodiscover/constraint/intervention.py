from typing import List, Optional, Tuple

import networkx as nx
import pandas as pd

from dodiscover._protocol import EquivalenceClass
from dodiscover.cd import BaseConditionalDiscrepancyTest
from dodiscover.ci import BaseConditionalIndependenceTest
from dodiscover.constraint import LearnInterventionSkeleton, SkeletonMethods
from dodiscover.context import Context
from dodiscover.typing import SeparatingSet

from .fcialg import FCI


class PsiFCI(FCI):
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
    cd_estimator : BaseConditionalDiscrepancyTest
        The conditional discrepancy test function.
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
    pds_skeleton_method : SkeletonMethods
        The method to use for learning the skeleton using PDS. Must be one of
        ('pds', 'pds_path'). See Notes for more details.
    known_intervention_targets : bool, optional
        If `True`, then will run the I-FCI algorithm. If `False`, will run the
        Psi-FCI algorithm. By default False.
    ci_estimator_kwargs : dict
        Keyword arguments for the ``ci_estimator`` function.
    cd_estimator_kwargs : dict
        Keyword arguments for the ``cd_estimator`` function.

    Notes
    -----
    Selection bias is unsupported because it is still an active research area.
    """

    def __init__(
        self,
        ci_estimator: BaseConditionalIndependenceTest,
        cd_estimator: BaseConditionalDiscrepancyTest,
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
        ci_estimator_kwargs=None,
        cd_estimator_kwargs=None,
    ):
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
            ci_estimator_kwargs=ci_estimator_kwargs,
        )
        self.cd_estimator = cd_estimator
        self.known_intervention_targets = known_intervention_targets
        self.cd_estimator_kwargs = cd_estimator_kwargs

    def learn_skeleton(
        self, data: pd.DataFrame, context: Context, sep_set: Optional[SeparatingSet] = None
    ) -> Tuple[nx.Graph, SeparatingSet]:
        # now compute all possibly d-separating sets and learn a better skeleton
        skel_alg = LearnInterventionSkeleton(
            self.ci_estimator,
            self.cd_estimator,
            sep_set=sep_set,
            alpha=self.alpha,
            min_cond_set_size=self.min_cond_set_size,
            max_cond_set_size=self.max_cond_set_size,
            max_combinations=self.max_combinations,
            skeleton_method=self.skeleton_method,
            second_stage_skeleton_method=self.pds_skeleton_method,
            keep_sorted=False,
            max_path_length=self.max_path_length,
            ci_estimator_kwargs=self.ci_estimator_kwargs,
            cd_estimator_kwargs=self.cd_estimator_kwargs,
        )
        skel_alg.fit(data, context)

        skel_graph = skel_alg.adj_graph_
        sep_set = skel_alg.sep_set_
        self.n_ci_tests += skel_alg.n_ci_tests
        return skel_graph, sep_set

    def fit(self, data: List[pd.DataFrame], context: Context):
        """Learn the relevant causal graph equivalence class.

        From the pairs of datasets, we take all combinations and
        construct F-nodes corresponding to those.

        Parameters
        ----------
        data : List[pd.DataFrame]
            The list of different datasets assigned to different
            environments. We assume the first dataset is always
            observational.
        context : Context
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if not isinstance(data, list):
            raise RuntimeError("The input datasets must be in a Python list.")

        n_datasets = len(data)
        intervention_targets = context.intervention_targets

        if n_datasets - 1 != len(intervention_targets):
            raise RuntimeError(
                f"There are {n_datasets} passed in, but {len(intervention_targets)} "
                f"intervention targets. There must be a matching (number of datasets - 1) and "
                f"intervention targets."
            )

        super().fit(data, context)

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
