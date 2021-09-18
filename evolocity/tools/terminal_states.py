from .. import settings
from .. import logging as logg
from ..preprocessing.neighbors import get_connectivities, verify_neighbors
from .transition_matrix import transition_matrix
from .utils import scale, groups_to_bool, strings_to_categoricals, get_plasticity_score

from scipy.sparse import linalg, csr_matrix, issparse
import numpy as np


def eigs(T, k=10, eps=1e-3, perc=None, random_state=None, v0=None):
    if random_state is not None:
        np.random.seed(random_state)
        v0 = np.random.rand(min(T.shape))
    try:
        # find k eigs with largest real part, and sort in descending order of eigenvals
        eigvals, eigvecs = linalg.eigs(T.T, k=k, which="LR", v0=v0)
        p = np.argsort(eigvals)[::-1]
        eigvals = eigvals.real[p]
        eigvecs = eigvecs.real[:, p]

        # select eigenvectors with eigenvalue of 1 - eps.
        idx = eigvals >= 1 - eps
        eigvals = eigvals[idx]
        eigvecs = np.absolute(eigvecs[:, idx])

        if perc is not None:
            lbs, ubs = np.percentile(eigvecs, perc, axis=0)
            eigvecs[eigvecs < lbs] = 0
            eigvecs = np.clip(eigvecs, 0, ubs)
            eigvecs /= eigvecs.max(0)

    except:
        eigvals, eigvecs = np.empty(0), np.zeros(shape=(T.shape[0], 0))

    return eigvals, eigvecs


def write_to_obs(adata, key, vals, node_subset=None):
    if node_subset is None:
        adata.obs[key] = vals
    else:
        vals_all = (
            adata.obs[key].copy() if key in adata.obs.keys() else np.zeros(adata.n_obs)
        )
        vals_all[node_subset] = vals
        adata.obs[key] = vals_all


def terminal_states(
    data,
    vkey="velocity",
    groupby=None,
    groups=None,
    self_transitions=False,
    eps=1e-3,
    random_state=0,
    exp_scale=50,
    copy=False,
    **kwargs,
):
    """Computes terminal states (root and end points).

    The end points and root nodes are obtained as stationary states of the
    velocity-inferred transition matrix and its transposed, respectively,
    which is given by left eigenvectors corresponding to an eigenvalue of 1, i.e.

    .. math::
        μ^{\\textrm{end}}=μ^{\\textrm{end}} \\pi, \quad
        μ^{\\textrm{root}}=μ^{\\textrm{root}} \\pi^{\\small \\textrm{T}}.

    .. code:: python

        evo.tl.terminal_states(adata)
        evo.pl.scatter(adata, color=['root_nodes', 'end_points'])

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
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
        Passed to evolocity.tl.transition_matrix(), e.g. basis, weight_diffusion.

    Returns
    -------
    Returns or updates `data` with the attributes
    root_nodes: `.obs`
        sparse matrix with transition probabilities.
    end_points: `.obs`
        sparse matrix with transition probabilities.
    """
    adata = data.copy() if copy else data
    verify_neighbors(adata)

    logg.info("computing terminal states", r=True)

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
        connectivities = get_connectivities(_adata, "distances")

        T_velo = data.uns[f'{vkey}_graph'] + data.uns[f'{vkey}_graph_neg']
        T_velo = np.expm1(T_velo * exp_scale)
        T_velo.data += 1
        T_velo = T_velo.T
        eigvecs_roots = eigs(T_velo, eps=eps, perc=[2, 98], random_state=random_state)[1]
        roots_velo = csr_matrix.dot(connectivities, eigvecs_roots).sum(1)

        T = transition_matrix(_adata, vkey=vkey, backward=True, **kwargs)
        eigvecs_roots = eigs(T, eps=eps, perc=[2, 98], random_state=random_state)[1]
        roots = csr_matrix.dot(connectivities, eigvecs_roots).sum(1)
        roots += roots_velo
        roots = scale(np.clip(roots, 0, np.percentile(roots, 98)))
        write_to_obs(adata, "root_nodes", roots, node_subset)

        T = transition_matrix(_adata, vkey=vkey, backward=False, **kwargs)
        eigvecs_ends = eigs(T, eps=eps, perc=[2, 98], random_state=random_state)[1]
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)
        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))
        write_to_obs(adata, "end_points", ends, node_subset)

        n_roots, n_ends = eigvecs_roots.shape[1], eigvecs_ends.shape[1]
        groups_str = f" ({groups})" if isinstance(groups, str) else ""
        roots_str = f"{n_roots} {'regions' if n_roots > 1 else 'region'}"
        ends_str = f"{n_ends} {'regions' if n_ends > 1 else 'region'}"

        logg.info(
            f"    identified {roots_str} of root nodes "
            f"and {ends_str} of end points {groups_str}."
        )

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added\n"
        "    'root_nodes', root nodes of Markov diffusion process (adata.obs)\n"
        "    'end_points', end points of Markov diffusion process (adata.obs)"
    )
    return adata if copy else None
