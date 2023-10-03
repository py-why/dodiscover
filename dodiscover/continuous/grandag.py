import warnings
from typing import Callable, Dict, List, Optional, Type, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from dodiscover.context import Context
from dodiscover.continuous.utils import TrExpScipy, is_acyclic
from dodiscover.toporder._base import CAMPruning
from dodiscover.toporder.utils import pns as _pns

EPSILON = 1e-8
STOP_CRIT_WIN = 100
H_TOLERANCE = 1e-8
OMEGA_LAM = 1e-4
OMEGA_MU = 0.9
MU_INIT = 1e-3
LAM_INIT = 0.0
EDGE_CLAMP_RANGE = 1e-4


class BaseModel(nn.Module):
    """Base class for the Gran-DAG and Gran-DAG++ algorithm.

    This class implements an NN for each variable :math:`X_j` and encodes the graph
    as a weighted adjacency matrix.

    Parameters
    ----------
    num_vars : int
        Number of variables (nodes) in the graph.
    num_layers : int
        Number of hidden layers in the NN.
    hid_dim: int
        Number of units in the hidden layers.
    num_params: int
        Number of parameters predicted by the NN.
    nonlin: Callable
        Nonlinearity to use as activation function for the NN.
    norm_prod_paths: bool
        Normalize the path product.
    square_prod: bool
        Use :math:`W^2` instead of :math:`|W|` to compute the path product.
    """

    def __init__(
        self,
        num_vars: int,
        num_layers: int,
        hid_dim: int,
        num_params: int,
        nonlin: Callable,
        norm_prod_paths: bool,
        square_prod: bool,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.num_params = num_params
        self.nonlin = nonlin
        self.norm_prod_paths = norm_prod_paths
        self.square_prod = square_prod

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.extra_params = (
            nn.ParameterList()
        )  # Parameters that are learned but independent of the parents

        # initialize current adjacency matrix
        self.register_buffer(
            "adjacency", torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars)
        )

        self.zero_weights_ratio = 0.0
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim
            if i == 0:
                in_dim = self.num_vars
            if i == self.num_layers:
                out_dim = self.num_params
            self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
            self.numel_weights += self.num_vars * out_dim * in_dim

        # Initialize weights with Xavier
        self._weight_init()

    def forward_given_params(
        self, x: torch.Tensor, weights: List[torch.Tensor], biases: List[torch.Tensor]
    ):
        """Run forward pass with given weights and biases.

        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_vars)
            Input tensor to the NN
        weights : list
            List of length num_layers+1 containing weights
        biases : list
            List of length num_layers+1 containing biases

        Returns
        -------
        output : tuple
            Tuple of length num_vars containing predicted parameters for each variable.
        """
        num_zero_weights = 0
        for k in range(self.num_layers + 1):
            # apply affine operator
            if k == 0:
                adj = self.adjacency.unsqueeze(0)
                x = torch.einsum("tij,ljt,bj->bti", weights[k], adj, x) + biases[k]
            else:
                x = torch.einsum("tij,btj->bti", weights[k], x) + biases[k]

            # count num of zeros
            num_zero_weights += weights[k].numel() - weights[k].nonzero().size(0)

            # apply non-linearity
            if k != self.num_layers:
                x = self.nonlin(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)
        return torch.unbind(x, 1)

    def get_w_adj(self):
        """Get the weighted adjacency matrix.

        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        Returns
        -------
        output : torch.Tensor of shape (num_vars, num_vars)
            Weighted adjacency matrix of the graph.
        """
        dev = self.adjacency.device
        dtype = self.adjacency.dtype
        weights = self.get_parameters(mode="w")[0]
        prod = torch.eye(self.num_vars).to(device=dev, dtype=dtype)
        if self.norm_prod_paths:
            prod_norm = torch.eye(self.num_vars).to(device=dev, dtype=dtype)
        for i, w in enumerate(weights):
            if self.square_prod:
                w = w**2
            else:
                w = torch.abs(w)
            if i == 0:
                prod = torch.einsum("tij,ljt,jk->tik", w, self.adjacency.unsqueeze(0), prod)
                if self.norm_prod_paths:
                    tmp = (1.0 - torch.eye(self.num_vars).unsqueeze(0)).to(device=dev, dtype=dtype)
                    prod_norm = torch.einsum(
                        "tij,ljt,jk->tik",
                        torch.ones_like(w).detach().to(device=dev, dtype=dtype),
                        tmp,
                        prod_norm,
                    )
            else:
                prod = torch.einsum("tij,tjk->tik", w, prod)
                if self.norm_prod_paths:
                    prod_norm = torch.einsum(
                        "tij,tjk->tik",
                        torch.ones_like(w).detach().to(device=dev, dtype=dtype),
                        prod_norm,
                    )

        # sum over density parameter axis
        prod = torch.sum(prod, 1)
        if self.norm_prod_paths:
            prod_norm = torch.sum(prod_norm, 1)
            denominator = prod_norm + torch.eye(self.num_vars).to(device=dev, dtype=dtype)
            return (prod / denominator).t()
        else:
            return prod.t()

    def _weight_init(self):
        """Initializes weights and biases of the NN.

        This function initializes the weights and biases of the NN according to Xavier
        uniform distribution (also known as Glorot initialization) :footcite:`glorot2010`.
        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        References
        ----------
        .. footbibliography::
        """
        with torch.no_grad():
            for node in range(self.num_vars):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain(self.nonlin.__name__))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self, mode="wbx"):
        """Get the requested parameters from the model.

        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        Parameters
        ----------
        mode : str
            Which parameters to get. if 'w' in mode, return weights, if 'b' mode return biases,
            if 'x' in mode return extra parameters

        Returns
        -------
        params : tuple
            Tuple of length len(mode) containing lists of the requested parameters.
        """

        params = []

        if "w" in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if "b" in mode:
            biases = []
            for b in self.biases:
                biases.append(b)
            params.append(biases)

        if "x" in mode:
            extra_params = []
            for extra_param in self.extra_params:
                if extra_param.requires_grad:
                    extra_params.append(extra_param)
            params.append(extra_params)

        return tuple(params)

    def compute_log_likelihood(self, x, weights, biases, extra_params, detach=False):
        """For some minibatch x return log-likelihood of the model.

        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_vars)
            Input tensor to the NN
        weights: list
            List containing the weights of the NN.
        biases: list
            List containing the biases of the NN.

        Returns
        -------
        log_probs : torch.Tensor of shape (batch_size, num_vars)
            Tensor containing the log likelihoods for each sample and variable
        """
        density_params = self.forward_given_params(x, weights, biases)

        if len(extra_params) != 0:
            extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []
        for i in range(self.num_vars):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
            conditional = self.get_distribution(density_param)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))
        return torch.cat(log_probs, 1)

    def get_distribution(self):
        raise NotImplementedError


class GranDAGModel(BaseModel):
    """
    Gran-DAG, where variances are learned (as extra parameters) but independent of the parents.
    """

    def __init__(
        self,
        num_vars: int,
        num_layers: int = 2,
        hid_dim: int = 10,
        nonlin: Callable = F.leaky_relu,
        norm_prod_paths: bool = True,
        square_prod: bool = False,
    ):
        super().__init__(
            num_vars,
            num_layers,
            hid_dim,
            num_params=1,
            nonlin=nonlin,
            norm_prod_paths=norm_prod_paths,
            square_prod=square_prod,
        )
        # extra parameters are log_std
        extra_params = np.ones((self.num_vars,))
        # each element in the list represents a variable, the size of the element is the number of
        # extra_params per var
        self.extra_params = nn.ParameterList()
        for extra_param in extra_params:
            self.extra_params.append(
                nn.Parameter(torch.tensor(np.log(extra_param).reshape(1)).type(torch.Tensor))
            )

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], dp[1])

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev


class GranDAGppModel(BaseModel):
    """
    Gran-DAG++, an extension of Gran-DAG, where variances are also learned through the NN.
    """

    def __init__(
        self,
        num_vars: int,
        num_layers: int = 2,
        hid_dim: int = 10,
        nonlin: Callable = F.leaky_relu,
        norm_prod_paths: bool = True,
        square_prod: bool = True,
    ):
        super().__init__(
            num_vars,
            num_layers,
            hid_dim,
            num_params=2,
            nonlin=nonlin,
            norm_prod_paths=norm_prod_paths,
            square_prod=square_prod,
        )

    def get_distribution(self, dp):
        return torch.distributions.normal.Normal(dp[0], torch.exp(dp[1]))


class GranDAG(CAMPruning):
    """The GranDAG (Gradient-based Neural DAG Learning) algorithm for causal discovery.

    GranDAG :footcite:`Lachapelle2020` extends NOTEARS :footcite:`zheng2018` to allow
    for nonlinear conditionals by using neural networks. Additionally, it employs preliminary
    neighbors selection (PNS) and CAM pruning :footcite:`Buhlmann2013`.

    Parameters
    ----------
    model: BaseModel
        Which model to use, default is GranDAGModel. GranDAGModel learns separate MLPs for
        each variable :math:`X_j` and predicts :math:`\mu_j` of the Gaussian distribution
        :math:`X_j | Pa_{j} \sim \mathcal{N}(\mu_j, \sigma^2_j)`, where :math:`Pa_{j}`
        denotes the parents of :math:`X_j` and :math:`\sigma^2_j` is also learned (but not
        dependent on :math:`Pa_{j}`. In GranDAGppModel (GraN-DAG++ in :footcite:`Lachapelle2020`)
        :math:`\sigma^2_j` are also predicted by the MLPs and thus dependent on the parents.
    val_ratio: float
        Fraction of data to use for validation, default is 0.2. The remaining data is
        used for training.
    mbs: int
        Mini-batch size to use, default is 64.
    lr_first_subproblem: float
        Learning rate for the first subproblems, default is 1e-4.
    lr: float
        Learning rate for all subsequent subproblems, default is 1e-2.
    num_train_iter: int
        Maximum number of iterations for the optimization procedure, default is 100000.
        Early  stopping is used if the algorithm converged before that number.
    prune : bool
        If True (default), apply CAM-pruning after finding the topological order.
    alpha : float, optional
        Alpha cutoff value for variable selection with hypothesis testing over regression
        coefficients, default is 0.01.
    n_splines : int, optional
        Number of splines to use for the feature function, default is 10.
        Automatically decreased in case of insufficient samples
    splines_degree: int, optional
        Order of spline to use for the feature function, default is 3.
    pns : bool
        If True, perform Preliminary Neighbour Search (PNS) before running the optimization
        procedure, default is True.
    pns_num_neighbors: int, optional
        Number of neighbors to use for PNS, default is 20.
    pns_threshold: float, optional
        Threshold to use for PNS, default is 1.
    double_prec: bool
        Whether to use double precision, default is True.
    cuda: bool
        Whether to use the GPU (if available), default is False.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Implementation is very similar to (and to a large extent taken from) the author's
    official GitHub implementation: https://github.com/kurowasan/GraN-DAG, licensed
    under the MIT license. This is also remarked for each function directly taken from
    aforementioned repository.
    Prior knowledge about the included and excluded directed edges in the output DAG
    is partially supported. If provided, the excluded edges are not initialized in the
    weighted adjacency matrix :math:`A_\\theta`. Included edges are not pruned in the
    final pruning step (if `prune=True`). Note: Included edges may still be removed
    during fitting of the algorithm. Including edges also raises a warning.
    For larger graphs computation time may be greatly reduced if run on a GPU. For
    this, set `cuda = True`. For smaller graphs, however, this may introduce additional
    overhead, thus increasing computation time over the CPU version.
    """

    def __init__(
        self,
        model: Union[Type[GranDAGModel], Type[GranDAGppModel]] = GranDAGModel,
        val_ratio: float = 0.2,
        mbs: int = 64,
        lr_first_subproblem: float = 1e-2,
        lr: float = 1e-4,
        num_train_iter: int = 100000,
        prune: bool = True,
        alpha: float = 0.001,
        n_splines: int = 10,
        splines_degree: int = 3,
        pns: bool = True,
        pns_num_neighbors: Optional[int] = 20,
        pns_threshold: Optional[float] = 1.0,
        double_prec: bool = True,
        cuda: bool = False,
    ):
        super().__init__(alpha, n_splines, splines_degree)
        self.model_class = model
        self.val_ratio = val_ratio
        self.mbs = mbs
        self.lr = lr
        self.lr_first_subproblem = lr_first_subproblem
        self.num_train_iter = num_train_iter
        self.pruning = prune
        self.pns = pns
        self.pns_num_neighbors = pns_num_neighbors
        self.pns_threshold = pns_threshold
        self.device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
        self.dtype = torch.float64 if double_prec else torch.float32

        # Attributes
        self.graph_: nx.DiGraph = nx.empty_graph()
        self.labels_to_nodes: Dict[Union[str, int], int] = dict()
        self.nodes_to_labels: Dict[int, Union[str, int]] = dict()

    def fit(self, data_df: pd.DataFrame, context: Context) -> None:
        """
        Fit Gran-DAG model to the data.

        Parameters
        ----------
        data_df : pd.DataFrame
            Datafame of the input data.
        context: Context
            The context of the causal discovery problem.
        """
        X = torch.tensor(data_df.to_numpy())
        self.context = context
        if len(self.context.included_edges.edges()) > 0:
            warnings.warn(
                "Prior knowledge is only partially supported for this algorithm. Included edges may"
                " be removed during fitting (but not during final pruning step)!"
            )

        # Data structure to exchange labels with nodes number
        self.nodes_to_labels = {i: data_df.columns[i] for i in range(len(data_df.columns))}
        self.labels_to_nodes = {data_df.columns[i]: i for i in range(len(data_df.columns))}

        # Check acyclicity condition on included_edges
        self._dag_check_included_edges()

        # Init model and optimizer
        self.model = self.model_class(num_vars=X.size(1)).to(device=self.device, dtype=self.dtype)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr_first_subproblem)

        # Exclude edges from the initial weighted adjacency matrix
        G_excluded = self._get_excluded_edges_graph()
        with torch.no_grad():
            for (i, j) in G_excluded.edges():
                self.model.adjacency[i, j] = 0.0

        # Apply preliminary neighborhood selection
        if self.pns:
            pns_num_neighbors = (
                X.size(1) if self.pns_num_neighbors is None else self.pns_num_neighbors
            )
            print(f"Run PNS with {pns_num_neighbors} neighbors and threshold {self.pns_threshold}")
            self.pns_(X, num_neighbours=pns_num_neighbors, threshold=self.pns_threshold)

        # Train model
        self._prepare_data(X)
        self._train()

        # Remove edges until we have DAG
        self._to_dag(X)

        # Further prune using CAM
        adj = self.model.adjacency.detach().cpu().numpy()
        if self.pruning:
            adj = self._prune(X, adj)

        self.graph_ = self._postprocess_output(nx.from_numpy_array(adj, create_using=nx.DiGraph))

    def _prepare_data(self, data: torch.Tensor) -> None:
        """Split data into train/val and normalize it.

        Parameters
        ----------
        data : torch.Tensor
            (N, K) matrix with N samples and K variables
        """
        X = data.detach().clone()
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)
        val_size = int(X.size(0) * self.val_ratio)
        train_size = X.size(0) - val_size
        X_train, X_val = X[:train_size], X[train_size:]
        sampler_train = RandomSampler(
            X_train, replacement=True, num_samples=self.num_train_iter * self.mbs
        )
        self.dataloader = {
            "train": DataLoader(X_train, min(X_train.size(0), self.mbs), sampler=sampler_train),
            "val": DataLoader(X_val, min(X_val.size(0), self.mbs)),
        }

    def _train(self):
        """Run Gran-DAG training"""
        mu = MU_INIT
        lam = LAM_INIT

        hs = []
        nlls = {"train": [], "val": []}
        aug_lagrangians = {"train": [], "val": []}

        for i_batch, batch in enumerate(tqdm(self.dataloader["train"], desc="Train model: ")):
            self.model.train()
            x = batch.to(device=self.device, dtype=self.dtype)
            weights, biases, extra_params = self.model.get_parameters(mode="wbx")
            loss = -torch.mean(self.model.compute_log_likelihood(x, weights, biases, extra_params))
            nlls["train"].append(loss.item())
            self.model.eval()

            # Constraint related
            w_adj = self.model.get_w_adj()
            h = self._compute_constraint(w_adj)

            # Compute augmented langrangian
            aug_lagrangian = loss + 0.5 * mu * h**2 + lam * h
            aug_lagrangians["train"].append(aug_lagrangian.item())

            # Optimization step on augmented lagrangian
            self.optimizer.zero_grad()
            aug_lagrangian.backward()
            self.optimizer.step()

            # Clamp edges
            with torch.no_grad():
                to_keep = w_adj > EDGE_CLAMP_RANGE
                self.model.adjacency *= to_keep

            # Compute loss on whole validation set
            if i_batch % STOP_CRIT_WIN == 0:
                loss_val = 0.0
                with torch.no_grad():
                    for batch in self.dataloader["val"]:
                        batch = batch.to(device=self.device, dtype=self.dtype)
                        loss_val += -torch.mean(
                            self.model.compute_log_likelihood(batch, weights, biases, extra_params)
                        ).item()
                loss_val /= len(self.dataloader["val"])
                aug_lagrangians["val"].append(
                    [i_batch, loss_val + aug_lagrangians["train"][-1] - nlls["train"][-1]]
                )

            # Compute delta for lambda
            if i_batch >= 2 * STOP_CRIT_WIN and i_batch % (2 * STOP_CRIT_WIN) == 0:
                t0, t_half, t1 = (
                    aug_lagrangians["val"][-3][1],
                    aug_lagrangians["val"][-2][1],
                    aug_lagrangians["val"][-1][1],
                )

                # If the validation loss went up and down, do not update lagrangian and penalty
                # coefficients.
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lam = -np.inf
                else:
                    delta_lam = (t1 - t0) / STOP_CRIT_WIN
            else:
                delta_lam = -np.inf  # do not update lambda nor mu

            # Check if the augmented lagrangian converged
            if h > H_TOLERANCE:
                if abs(delta_lam) < OMEGA_LAM or delta_lam > 0:
                    lam += mu * h.item()
                    # Did the constraint improve sufficiently?
                    hs.append(h.item())
                    if len(hs) >= 2:
                        if hs[-1] > hs[-2] * OMEGA_MU:
                            mu *= 10

                    # Reinitialize learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.lr
            else:
                # Final clamping of edges
                with torch.no_grad():
                    to_keep = w_adj > 0
                    self.model.adjacency *= to_keep
                break
        else:
            warnings.warn(
                f"h not converged after {self.num_train_iter} iterations (h: {h}, h_thres: "
                f"{H_TOLERANCE})!"
            )
            # Final clamping of edges
            with torch.no_grad():
                to_keep = w_adj > 0
                self.model.adjacency *= to_keep

    def _to_dag(self, X: torch.Tensor) -> None:
        """Remove edges until we obtain a DAG

        Compute the average Jacobian matrix :math:`\mathcal{J}_{i,j}` of the lagrangian over
        all samples of the dataset. Then, remove edges :math:`(i,j)` for which
        :math:`\mathcal{J}_{i,j}` is the lowest until the adjacency matrix represents a DAG.

        Parameters
        ----------
        X : torch.Tensor of shape (num_samples, num_vars)
            Tensor containing all samples.
        """
        self.model.eval()
        A = self._compute_jacobian_avg(X).t().detach().cpu().numpy()

        with torch.no_grad():
            # Find the smallest threshold that removes all cycle-inducing edges
            thresholds = np.unique(A)
            for step, t in enumerate(thresholds):
                to_keep = torch.Tensor(A > t + EPSILON).to(device=self.device, dtype=self.dtype)
                new_adj = self.model.adjacency * to_keep

                if is_acyclic(new_adj):
                    self.model.adjacency.copy_(new_adj)
                    break

    def _dag_check_included_edges(self) -> None:
        """Check that the edges included in `self.context` does not violate DAG condition."""
        is_dag = nx.is_directed_acyclic_graph(self._get_included_edges_graph())
        if nx.is_empty(self._get_included_edges_graph()):
            is_dag = True
        if not is_dag:
            raise ValueError("Edges included in the graph violate the acyclicity condition!")

    def _postprocess_output(self, graph):
        """Relabel the graph nodes with the custom labels of the input dataframe.

        Parameters
        ----------
        graph : nx.DiGraph
            Networkx directed graph with nodes to relabel.

        Returns
        -------
        G : nx.DiGraph
            Graph with the relabeled nodes.
        """
        G = nx.relabel_nodes(graph, mapping=self.nodes_to_labels)
        return G

    def _get_included_edges_graph(self):
        """Get the `self.context.included_edges` graph with numerical label of the nodes.

        The returned directed graph of the included edges, with one node for each column
        in the dataframe input of the `fit` method, and numerical label of the nodes.

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph of the edges included in `self.context`.
        """
        num_nodes = len(self.labels_to_nodes)
        G = nx.empty_graph(n=num_nodes, create_using=nx.DiGraph)
        for edge in self.context.included_edges.edges():
            u, v = self.labels_to_nodes[edge[0]], self.labels_to_nodes[edge[1]]
            G.add_edge(u, v)
        return G

    def _get_excluded_edges_graph(self):
        """Get the `self.context.excluded_edges` graph with numerical label of the nodes.

        The returned directed graph of the excluded edges, with one node for each column
        in the dataframe input of the `fit` method, and numerical label of the nodes.

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph of the edges excluded in `self.context`.
        """
        num_nodes = len(self.labels_to_nodes)
        G = nx.empty_graph(n=num_nodes, create_using=nx.DiGraph)
        for edge in self.context.excluded_edges.edges():
            u, v = self.labels_to_nodes[edge[0]], self.labels_to_nodes[edge[1]]
            G.add_edge(u, v)
        return G

    def pns_(self, X, num_neighbours, threshold):
        """Wrapper around dodiscover.toporder.utils.pns using the adjacency matrix of the model"""
        model_adj = self.model.adjacency.detach().cpu().numpy()
        model_adj = _pns(model_adj, X, threshold, num_neighbours)

        with torch.no_grad():
            self.model.adjacency.copy_(torch.Tensor(model_adj))

    def _prune(
        self,
        X: torch.tensor,
        A: NDArray,
    ) -> NDArray:
        """Prune the adjacency matrix `A` that is returned by the model from spurious edges.

        Use sparse regression over the matrix of the data `X` for variable selection over the
        edges in the adjacency matrix `A`

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, num_vars)
            Matrix of the data.
        A : np.ndarray of shape (n_nodes, n_nodes)
            Adjacency matrix to be pruned.

        Returns
        -------
        A : np.ndarray
            The pruned adjacency matrix output of the causal discovery algorithm.
        """
        G_included = self._get_included_edges_graph()
        G_excluded = self._get_excluded_edges_graph()

        _, d = X.shape
        X = X.detach().cpu()
        A_pruned = np.zeros((d, d))
        order = list(nx.topological_sort(nx.from_numpy_array(A, create_using=nx.DiGraph)))
        for c in range(d):
            pot_parents = []
            for p in order[: order.index(c)]:
                if ((not G_excluded.has_edge(p, c)) and A[p, c] == 1) or G_included.has_edge(p, c):
                    pot_parents.append(p)
            if len(pot_parents) > 0:
                parents = self._variable_selection(
                    X[:, pot_parents].numpy(), X[:, c].numpy(), pot_parents, c, G_included
                )
                for parent in parents:
                    A_pruned[parent, c] = 1

        return A_pruned

    def _compute_jacobian_avg(self, X):
        """Compute average Jacobian over the data X.

        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        Parameters
        ----------
        X : torch.Tensor of shape (n_samples, num_vars)
            Matrix of the data.

        Returns
        -------
        jac_avg : torch.tensor of shape (num_vars, num_vars)
            Average Jacobian of the Lagrangian.
        """
        jac_avg = torch.zeros(self.model.num_vars, self.model.num_vars).to(
            device=self.device, dtype=self.dtype
        )

        x = X.detach().clone().to(device=self.device, dtype=self.dtype).requires_grad_()

        # compute loss
        weights, biases, extra_params = self.model.get_parameters(mode="wbx")
        log_probs = self.model.compute_log_likelihood(x, weights, biases, extra_params, detach=True)
        log_probs = torch.unbind(log_probs, 1)

        # compute jacobian of the loss
        for i in range(self.model.num_vars):
            tmp = torch.autograd.grad(
                log_probs[i],
                x,
                retain_graph=True,
                grad_outputs=torch.ones(x.size(0)).to(device=self.device, dtype=self.dtype),
            )[0]
            jac_avg[i, :] = torch.abs(tmp).mean(0)

        return jac_avg

    def _compute_constraint(self, w_adj):
        """Compute acyclicity constraint of the weighted adjacency matrix.

        From the author's GitHub implementation: https://github.com/kurowasan/GraN-DAG.

        Parameters
        ----------
        w_adj : torch.tensor of shape (num_vars, num_vars)
            Weighted adjacency matrix

        Returns
        -------
        h : torch.Tensor
            Acyclicity constraint
        """
        if (w_adj < 0).detach().cpu().numpy().any():
            raise ValueError("Found negative values in weighted adjacency matrix!")
        h = TrExpScipy.apply(w_adj) - self.model.num_vars
        return h
