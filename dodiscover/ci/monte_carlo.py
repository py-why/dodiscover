from typing import Optional

import numpy as np
import scipy.spatial
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors


def generate_knn_in_subspace(
    z_arr: ArrayLike, method: str = "knn", k: int = 1, n_jobs: Optional[int] = None
) -> ArrayLike:
    """Generate kNN in subspace.

    Parameters
    ----------
    z_arr : ArrayLike of shape (n_samples, n_features_z)
        The covariate space.
    method : str, optional
        Method to use, by default 'knn'. Can be ('knn', 'kdtree').
    k : int, optional
        The number of k-nearest neighbors to query, by default 1.
    n_jobs : int,
        The number of CPUs to use for joblib. By default, None.

    Returns
    -------
    indices : ArrayLike of shape (n_samples, k)
        The indices of the k-nearest-neighbors for each sample.
    """
    # use a method to generate k-nearest-neighbors in subspace of Z
    if method == "knn":
        # compute the nearest neighbors in the space of "Z training" using ball-tree alg.
        nbrs = NearestNeighbors(
            n_neighbors=k + 1, algorithm="ball_tree", metric="l2", n_jobs=n_jobs
        ).fit(z_arr)
        # then get the K nearest nbrs in the Z space
        _, indices = nbrs.kneighbors(z_arr)
    elif method == "kdtree":
        tree_xyz = scipy.spatial.cKDTree(z_arr)
        indices = tree_xyz.query(z_arr, k=k, p=np.inf, eps=0.0, workers=n_jobs)[1].astype(np.int32)

    return indices


def restricted_nbr_permutation(nbrs: ArrayLike, random_seed=None) -> ArrayLike:
    """Compute a permutation of neighbors with restrictions.

    Parameters
    ----------
    nbrs : ArrayLike of shape (n_samples, k)
        The k-nearest-neighbors for each sample index.
    random_seed : int, optional
        Random seed, by default None.

    Returns
    -------
    restricted_perm : ArrayLike of shape (n_samples)
        The final permutation order of the sample indices. There may be
        repeating samples. See Notes for details.

    Notes
    -----
    Restricted permutation goes through random samples and looks at the k-nearest
    neighbors (columns of ``nbrs``) and shuffles the closest neighbor index only
    if it has not been used to permute another sample. If it has been, then the
    algorithm looks at the next nearest-neighbor and so on. If all k-nearest
    neighbors of a sample has been checked, then a random neighbor is chosen. In this
    manner, the algorithm tries to perform permutation without replacement, but
    if necessary, will choose a repeating neighbor sample.
    """
    n_samples, k_dims = nbrs.shape
    rng = np.random.default_rng(seed=random_seed)

    # initialize the final permutation order
    restricted_perm = np.zeros((n_samples,))

    # generate a random order of samples to go through
    random_order = rng.permutation(n_samples)

    # keep track of values we have already used
    used = set()

    # go through the random order
    for idx in random_order:
        m = 0
        use_idx = nbrs[idx, m]

        # if the current nbr is already used, continue incrementing
        # until we have either found a new sample to use, or if
        # we have reach the maximum number of shuffles to consider
        while (use_idx in used) and (m < k_dims - 1):
            m += 1
            use_idx = nbrs[idx, m]

        # check whether or not we have exhaustively checked all kNN
        if use_idx in used and m == k_dims:
            # XXX: Note this step is not in the original paper
            # choose a random neighbor to permute
            restricted_perm[idx] = rng.choice(nbrs[idx, :], size=1)
        else:
            # permute with the existing neighbor
            restricted_perm[idx] = use_idx
        used.add(use_idx)
    return restricted_perm
