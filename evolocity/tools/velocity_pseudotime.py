from .terminal_states import terminal_states
from .utils import groups_to_bool, scale, strings_to_categoricals

import numpy as np
from scanpy.tools._dpt import DPT
from scipy.sparse import coo_matrix, issparse, spdiags, linalg
import scipy.stats as ss

class VPT(DPT):
    def set_iroots(self, root=None):
        if (
            isinstance(root, str)
            and root in self._adata.obs.keys()
            and self._adata.obs[root].max() != 0
        ):
            self.iroots = np.array(self._adata.obs[root])
            self.iroots = scale(self.iroots)
            self.iroots = np.argwhere(
                self.iroots >= self.iroots.max()
            ).ravel()
        elif isinstance(root, str) and root in self._adata.obs_names:
            self.iroots = [ self._adata.obs_names.get_loc(root) ]
        elif isinstance(root, (int, np.integer)) and root < self._adata.n_obs:
            self.iroots = [ root ]
        else:
            self.iroots = [ None ]

    def compute_transitions(self, density_normalize=True):
        T = self._connectivities
        if density_normalize:
            q = np.asarray(T.sum(axis=0))
            q += q == 0
            Q = (
                spdiags(1.0 / q, 0, T.shape[0], T.shape[0])
                if issparse(T)
                else np.diag(1.0 / q)
            )
            K = Q.dot(T).dot(Q)
        else:
            K = T
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        Z = (
            spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
            if issparse(K)
            else np.diag(1.0 / z)
        )
        self._transitions_sym = Z.dot(K).dot(Z)

    def compute_eigen(self, n_comps=10, sym=None, sort='decrease'):
        if self._transitions_sym is None:
            raise ValueError('Run `.compute_transitions` first.')
        n_comps = min(self._transitions_sym.shape[0] - 1, n_comps)
        evals, evecs = linalg.eigsh(self._transitions_sym, k=n_comps, which='LM')
        self._eigen_values = evals[::-1]
        self._eigen_basis = evecs[:, ::-1]

    def compute_pseudotime(self, inverse=False):
        if self.iroot is not None:
            self._set_pseudotime()
            self.pseudotime = 1 - self.pseudotime if inverse else self.pseudotime
            self.pseudotime[~np.isfinite(self.pseudotime)] = np.nan
        else:
            self.pseudotime = np.empty(self._adata.n_obs)
            self.pseudotime[:] = np.nan

def velocity_pseudotime(
        adata,
        vkey='velocity',
        rank_transform=True,
        groupby=None,
        groups=None,
        root_key=None,
        end_key=None,
        use_ends=False,
        n_dcs=10,
        use_velocity_graph=True,
        save_diffmap=None,
        return_model=None,
        **kwargs,
):
    """Computes pseudotime based on the evolocity graph.

    Velocity pseudotime is a random-walk based distance measures on the velocity graph.
    After computing a distribution over root cells obtained from the velocity-inferred
    transition matrix, it measures the average number of steps it takes to reach a cell
    after start walking from one of the root cells. Contrarily to diffusion pseudotime,
    it implicitly infers the root cells and is based on the directed velocity graph
    instead of the similarity-based diffusion kernel.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    rank_transform: `bool` (default: `True`)
        Perform final rank transformation.
    groupby: `str`, `list` or `np.ndarray` (default: `None`)
        Key of observations grouping to consider.
    groups: `str`, `list` or `np.ndarray` (default: `None`)
        Groups selected to find terminal states on. Must be an element of
        adata.obs[groupby]. Only to be set, if each group is assumed to have a distinct
        lineage with an independent root and end point.
    root_key: `int` (default: `None`)
        Index of root cell to be used.
        Computed from velocity-inferred transition matrix if not specified.
    end_key: `int` (default: `None`)
        Index of end point to be used.
        Computed from velocity-inferred transition matrix if not specified.
    n_dcs: `int` (default: 10)
        The number of diffusion components to use.
    use_velocity_graph: `bool` (default: `True`)
        Whether to use the velocity graph.
        If False, it uses the similarity-based diffusion kernel.
    save_diffmap: `bool` (default: `None`)
        Whether to store diffmap coordinates.
    return_model: `bool` (default: `None`)
        Whether to return the vpt object for further inspection.
    **kwargs:
        Further arguments to pass to VPT (e.g. min_group_size, allow_kendall_tau_shift).

    Returns
    -------
    Updates `adata` with the attributes
    velocity_pseudotime: `.obs`
        Velocity pseudotime obtained from velocity graph.
    """

    strings_to_categoricals(adata)
    if root_key is None and 'root_nodes' in adata.obs.keys():
        root0 = adata.obs['root_nodes'][0]
        if not np.isnan(root0) and not isinstance(root0, str):
            root_key = 'root_nodes'
    if end_key is None and 'end_points' in adata.obs.keys():
        end0 = adata.obs['end_points'][0]
        if not np.isnan(end0) and not isinstance(end0, str):
            end_key = 'end_points'

    groupby = (
        'node_fate' if groupby is None and 'node_fate' in adata.obs.keys()
        else groupby
    )
    categories = (
        adata.obs[groupby].cat.categories
        if groupby is not None and groups is None
        else [None]
    )
    for cat in categories:
        groups = cat if cat is not None else groups
        if (
            root_key is None
            or root_key in adata.obs.keys()
            and np.max(adata.obs[root_key]) == np.min(adata.obs[root_key])
        ):
            terminal_states(adata, vkey, groupby, groups)
            root_key, end_key = 'root_nodes', 'end_points'
        node_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        data = adata.copy() if node_subset is None else adata[node_subset].copy()
        if 'allow_kendall_tau_shift' not in kwargs:
            kwargs['allow_kendall_tau_shift'] = True
        vpt = VPT(data, n_dcs=n_dcs, **kwargs)

        if use_velocity_graph:
            T = data.uns[f'{vkey}_graph'] - data.uns[f'{vkey}_graph_neg']
            vpt._connectivities = T + T.T

        vpt.compute_transitions()
        vpt.compute_eigen(n_comps=n_dcs)

        vpt.set_iroots(root_key)
        pseudotimes = [ np.zeros(adata.X.shape[0]) ]
        for iroot in vpt.iroots:
            if iroot is None:
                continue
            vpt.iroot = iroot
            vpt.compute_pseudotime()
            pseudotimes.append(scale(vpt.pseudotime))

        if use_ends:
            vpt.set_iroots(end_key)
            for iroot in vpt.iroots:
                if iroot is None:
                    continue
                vpt.iroot = iroot
                vpt.compute_pseudotime(inverse=True)
                pseudotimes.append(scale(vpt.pseudotime))

        vpt.pseudotime = np.nan_to_num(np.vstack(pseudotimes)).mean(0)
        if rank_transform:
            vpt.pseudotime = ss.rankdata(vpt.pseudotime)
        vpt.pseudotime = scale(vpt.pseudotime)

        if 'n_branchings' in kwargs and kwargs['n_branchings'] > 0:
            vpt.branchings_segments()
        else:
            vpt.indices = vpt.pseudotime.argsort()

        if f'{vkey}_pseudotime' not in adata.obs.keys():
            pseudotime = np.empty(adata.n_obs)
            pseudotime[:] = np.nan
        else:
            pseudotime = adata.obs[f'{vkey}_pseudotime'].values
        pseudotime[node_subset] = vpt.pseudotime
        adata.obs[f'{vkey}_pseudotime'] = np.array(pseudotime, dtype=np.float64)

        if save_diffmap:
            diffmap = np.empty(shape=(adata.n_obs, n_dcs))
            diffmap[:] = np.nan
            diffmap[node_subset] = vpt.eigen_basis
            adata.obsm[f'X_diffmap_{groups}'] = diffmap

    return vpt if return_model else None
