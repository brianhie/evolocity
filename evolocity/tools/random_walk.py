from .. import settings
from .. import logging as logg
from ..preprocessing.neighbors import verify_neighbors
from .transition_matrix import transition_matrix
from .utils import groups_to_bool, strings_to_categoricals

import numpy as np
import scipy.special


def random_walk(
    data,
    root_node=0,
    walk_length=10,
    n_walks=1,
    forward_walk=True,
    path_key='rw_paths',
    vkey='velocity',
    groupby=None,
    groups=None,
    self_transitions=False,
    eps=1e-3,
    random_state=0,
    copy=False,
    **kwargs,
):
    """Runs a random walk on the evolocity graph.

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    root_node: `int` (default: `0`)
        Index of node at which to start random walk.
    walk_length: `int` (default: `10`)
        Number of steps in walk.
    n_walks: `int` (default: `1`)
        Number of walks to take.
    forward_walk: `bool` (default: `True`)
        Whether to go in the same or reverse direction of evolocity.
    path_key: `str` (default: `'rw_paths'`)
        Name at which to store the random walks.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    groupby: `str`, `list` or `np.ndarray` (default: `None`)
        Key of observations grouping to consider. Only to be set, if each group is
        assumed to have a distinct lineage with an independent root and end point.
    groups: `str`, `list` or `np.ndarray` (default: `None`)
        Groups selected to find terminal states on. Must be an element of .obs[groupby].
        To be specified only for very distinct/disconnected clusters.
    self_transitions: `bool` (default: `False`)
        Allow transitions from one node to itself.
    eps: `float` (default: 1e-3)
        Tolerance for eigenvalue selection.
    random_state: `int` or None (default: 0)
        Seed used by the random number generator.
        If `None`, use the `RandomState` instance by `np.random`.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to data.
    **kwargs:
        Passed to evolocity.tl.transition_matrix(), e.g. scale, basis.

    Returns
    -------
    Returns or updates `data` with the attributes
    rw_paths: `.uns`
        rows of matrix correspond to random walks, columns correspond to steps
    """
    adata = data.copy() if copy else data
    verify_neighbors(adata)

    logg.info("running random walks", r=True)

    strings_to_categoricals(adata)
    if groupby is not None:
        logg.warn(
            "Only set groupby, when you have evident distinct clusters/lineages,"
            " each with an own root and end point."
        )

    kwargs.update({"self_transitions": self_transitions})
    categories = [None]
    if groupby is not None and groups is None:
        categories = adata.obs[groupby].cat.categories
    for cat in categories:
        groups = cat if cat is not None else groups
        node_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        _adata = adata if groups is None else adata[node_subset]

        if not node_subset[root_node]:
            logg.warn(
                "Root node not in grouped subset, skipping."
            )
            continue
        root_node_group = sum(node_subset[:root_node])

        T = transition_matrix(_adata, vkey=vkey, backward=(not forward_walk), **kwargs)
        n_nodes = _adata.X.shape[0]
        assert(T.shape[0] == T.shape[1] == n_nodes)
        paths = np.zeros((n_walks, walk_length + 1))
        paths[:, 0] = root_node_group

        for t in range(walk_length):
            path = []
            for w in range(n_walks):
                prob = scipy.special.softmax(T[paths[w, t], :].toarray().ravel())
                path.append(np.random.choice(n_nodes, p=prob))
            paths[:, t + 1] = path

        group_map = np.argwhere(node_subset).ravel()
        paths = np.vstack([
            group_map[np.array(paths[w], dtype=np.int32)] for w in range(n_walks)
        ])

        group_path_key = path_key
        if cat is not None:
            group_path_key += f'_{cat}'
        adata.uns[group_path_key] = paths

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n"
        f"    '{path_key}', random walk paths (adata.uns)\n"
    )
    return adata if copy else None
