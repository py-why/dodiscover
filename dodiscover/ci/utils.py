import numpy as np
from numpy.typing import NDArray


def _restricted_permutation(
    nbrs: NDArray[np.int], n_shuffle_nbrs: int, n_samples: int, random_state=None
) -> NDArray:
    if random_state is None:
        random_state = np.random.RandomState(seed=random_state)

    # initialize the final permutation order
    restricted_perm = np.zeros((n_samples,))

    # generate a random order of samples to go through
    random_order = random_state.permutation(n_samples)

    # keep track of values we have already used
    used = set()

    # go through the random order
    for idx in random_order:
        m = 0
        use_idx = nbrs[idx, m]

        # if the current nbr is already used, continue incrementing
        # until we have either found a new sample to use, or if
        # we have reach the maximum number of shuffles to consider
        while (use_idx in used) and (m < n_shuffle_nbrs - 1):
            m += 1
            use_idx = nbrs[idx, m]

        restricted_perm[idx] = use_idx
        used.add(use_idx)
    return restricted_perm
