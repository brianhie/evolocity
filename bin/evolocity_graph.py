from Bio import pairwise2, SeqIO
from Bio.SubsMat import MatrixInfo as matlist
from scanpy.tools._dpt import DPT
from scipy.sparse import coo_matrix, issparse, spdiags, linalg
import scvelo as scv
import scvelo.plotting.utils as scvu
from scvelo.preprocessing.moments import get_connectivities
from scvelo.preprocessing.neighbors import neighbors, verify_neighbors
from scvelo.preprocessing.neighbors import get_neighs, get_n_neighs
from scvelo.tools.utils import groups_to_bool, scale, strings_to_categoricals
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from mutation import predict_sequence_prob
from utils import *

def sum_var(A):
    """summation over axis 1 (var) equivalent to np.sum(A, 1)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.sum(1).A1 if issparse(A) else np.sum(A, axis=1)

def norm(A):
    """computes the L2-norm along axis 1
    (e.g. genes or embedding dimensions) equivalent to np.linalg.norm(A, axis=1)
    """
    if issparse(A):
        return np.sqrt(A.multiply(A).sum(1).A1)
    else:
        return np.sqrt(np.einsum('ij, ij -> i', A, A)
                       if A.ndim > 1 else np.sum(A * A))

def get_iterative_indices(
        indices,
        index,
        n_recurse_neighbors=0,
        max_neighs=None,
):
    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices

def get_indices(dist, n_neighbors=None, mode_neighbors='distances'):
    from scvelo.preprocessing.neighbors import compute_connectivities_umap

    D = dist.copy()
    D.data += 1e-6

    n_counts = sum_var(D > 0)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == 'distances':
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == 'connectivities':
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D

def likelihood_compare(seq1, seq2, args, vocabulary, model,
                       pos1=None, pos2=None, seq_cache={}, verbose=False):
    likelihoods = []

    for seq_pred, positions in zip([ seq1, seq2 ], [ pos1, pos2 ]):
        if positions is None:
            positions = range(len(seq_pred))

        if seq_pred in seq_cache:
            seq_probs = seq_cache[seq_pred][list(positions)]

        else:
            y_pred = predict_sequence_prob(
                args, seq_pred, vocabulary, model, verbose=verbose
            )
            if not ('esm' in args.model_name or args.model_name == 'tape'):
                y_pred = np.log(y_pred)

            seq_probs = np.array([
                y_pred[i + 1, (
                    vocabulary[seq_pred[i]]
                    if seq_pred[i] in vocabulary else
                    model.unk_idx_
                )]
                for i in positions
            ])

        likelihoods.append(np.mean(seq_probs))

    return likelihoods[1] - likelihoods[0]

def likelihood_full(seq1, seq2, args, vocabulary, model,
                    seq_cache={}, verbose=False):
    return likelihood_compare(
        seq1, seq2, args, vocabulary, model,
        seq_cache=seq_cache, verbose=verbose,
    )

def likelihood_muts(seq1, seq2, args, vocabulary, model,
                    seq_cache={}, verbose=False):
    # Align, prefer matches to gaps.
    alignment = pairwise2.align.globalms(
        seq1, seq2, 5, -4, -4, -.1, one_alignment_only=True
    )[0]
    a_seq1, a_seq2, _, _, _ = alignment

    # Map alignment to original indices.
    del1, sub1, del2, sub2 = [], [], [], []
    for a_seq, other_seq, deletions, substitutions in zip(
            [ a_seq1, a_seq2, ], [ a_seq2, a_seq1, ],
            [ del1, del2 ], [ sub1, sub2, ]
    ):
        orig_idx = 0
        for a_idx, ch in enumerate(a_seq):
            if ch == '-':
                continue
            if other_seq[a_idx] == '-':
                deletions.append(orig_idx)
            elif other_seq[a_idx] != ch:
                substitutions.append(orig_idx)
            orig_idx += 1

    return likelihood_compare(
        seq1, seq2, args, vocabulary, model,
        pos1=sub1, pos2=sub2, seq_cache=seq_cache, verbose=verbose,
    )

def likelihood_blosum62(
        seq1, seq2, args, vocabulary, model,
        seq_cache={}, verbose=False, natural_aas=None,
):
    from Bio.SubsMat import MatrixInfo as matlist
    matrix = matlist.blosum62

    # Align, prefer matches to gaps.
    alignment = pairwise2.align.globalms(
        seq1, seq2, 5, -4, -4, -.1, one_alignment_only=True
    )[0]
    a_seq1, a_seq2, _, _, _ = alignment

    scores = []
    for ch1, ch2 in zip(a_seq1, a_seq2):
        if ch1 == ch2:
            continue
        if (ch1, ch2) in matrix:
            scores.append(matrix[(ch1, ch2)])
        elif (ch2, ch1) in matrix:
            scores.append(matrix[(ch2, ch1)])

    return np.mean(scores)

def likelihood_self(seq1, seq2, args, vocabulary, model,
                    seq_cache={}, verbose=False):
    # Align, prefer matches to gaps.
    alignment =pairwise2.align.globalms(
        seq1, seq2, 5, -4, -4, -.1, one_alignment_only=True
    )[0]
    a_seq1, a_seq2, _, _, _ = alignment

    # See how mutating to `seq2' changes probability.

    likelihood_change = []

    for a_seq, other_seq in zip([ a_seq1, a_seq2 ],
                                [ a_seq2, a_seq1 ]):
        if a_seq in seq_cache:
            y_pred = seq_cache[a_seq]
        else:
            y_pred = predict_sequence_prob(
                args, a_seq, vocabulary, model, verbose=verbose
            )
            if not ('esm' in args.model_name or args.model_name == 'tape'):
                y_pred = np.log(y_pred)

        orig_idx, scores = 0, []
        for a_idx, ch in enumerate(a_seq):
            if ch == '-':
                continue
            if other_seq[a_idx] == '-':
                pass
            elif other_seq[a_idx] != ch:
                ch_idx = vocabulary[ch] \
                         if ch in vocabulary else \
                         model.unk_idx_
                o_idx = vocabulary[other_seq[a_idx]] \
                        if other_seq[a_idx] in vocabulary else \
                        model.unk_idx_

                prob_wt = y_pred[a_idx + 1, ch_idx]
                prob_mut = y_pred[a_idx + 1, o_idx]
                scores.append(prob_wt - prob_mut)
            orig_idx += 1

        likelihood_change.append(np.mean(scores))

    return likelihood_change[1] - likelihood_change[0]

def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()


class VelocityGraph:
    def __init__(
            self,
            adata,
            seqs,
            score='other',
            scale_dist=False,
            vkey='velocity',
            n_recurse_neighbors=None,
            random_neighbors_at_max=None,
            mode_neighbors='distances',
            verbose=False,
    ):
        self.adata = adata

        self.seqs = seqs
        self.seq_probs = {}

        self.score = score

        self.scale_dist = scale_dist

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if mode_neighbors == 'connectivities':
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if np.min((get_neighs(adata, 'distances') > 0).sum(1).A1) == 0:
            raise ValueError(
                'Your neighbor graph seems to be corrupted. '
                'Consider recomputing via scanpy.pp.neighbors.'
            )
        self.indices = get_indices(
            dist=get_neighs(adata, 'distances'),
            mode_neighbors=mode_neighbors,
        )[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f'{vkey}_graph', f'{vkey}_graph_neg'
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        self.self_prob = None

        self.verbose = verbose


    def compute_likelihoods(self, args, vocabulary, model):
        if self.verbose:
            iterator = tqdm(self.seqs)
        else:
            iterator = self.seqs

        for seq in iterator:
            y_pred = predict_sequence_prob(
                args, seq, vocabulary, model, verbose=self.verbose
            )

            if self.score == 'other':
                self.seq_probs[seq] = np.array([
                    y_pred[i + 1, (
                        vocabulary[seq[i]]
                        if seq[i] in vocabulary else
                        model.unk_idx_
                    )] for i in range(len(seq))
                ])
            else:
                self.seq_probs[seq] = y_pred

        if self.verbose:
            sys.stdout.flush()

    def compute_gradients(self, args, vocabulary, model):
        n_obs = self.adata.X.shape[0]
        vals, rows, cols, uncertainties = [], [], [], []

        if self.verbose:
            iterator = tqdm(range(n_obs))
        else:
            iterator = range(n_obs)

        for i in iterator:
            neighs_idx = get_iterative_indices(
                self.indices, i, self.n_recurse_neighbors, self.max_neighs
            )

            if self.score == 'other':
                score_fn = likelihood_muts
            else:
                score_fn = likelihood_self

            val = np.array([
                score_fn(
                    self.seqs[i], self.seqs[j],
                    args, vocabulary, model,
                    seq_cache=self.seq_probs, verbose=self.verbose,
                ) for j in neighs_idx
            ])

            if self.scale_dist:
                dist = self.adata.X[neighs_idx] - self.adata.X[i, None]
                dist = np.sqrt((dist ** 2).sum(1))
                val *= self.scale_dist * dist

            vals.extend(val)
            rows.extend(np.ones(len(neighs_idx)) * i)
            cols.extend(neighs_idx)

        if self.verbose:
            sys.stdout.flush()

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

def velocity_graph(
        adata,
        args,
        vocabulary,
        model,
        score='other',
        scale_dist=False,
        seqs=None,
        vkey='velocity',
        n_recurse_neighbors=0,
        random_neighbors_at_max=None,
        mode_neighbors='distances',
        copy=False,
        verbose=True,
):
    adata = adata.copy() if copy else adata
    verify_neighbors(adata)

    if seqs is None:
        seqs = adata.obs['seq']
    if adata.X.shape[0] != len(seqs):
        raise ValueError('Number of sequences should correspond to '
                         'number of observations.')

    valid_scores = { 'self', 'other' }
    if score not in valid_scores:
        raise ValueError('Score must be one of {}'
                         .format(', '.join(valid_scores)))

    vgraph = VelocityGraph(
        adata,
        seqs,
        score=score,
        scale_dist=scale_dist,
        vkey=vkey,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        mode_neighbors=mode_neighbors,
        verbose=verbose,
    )

    if verbose:
        tprint('Computing likelihoods...')
    vgraph.compute_likelihoods(args, vocabulary, model)
    if verbose:
        print('')

    if verbose:
        tprint('Computing velocity graph...')
    vgraph.compute_gradients(args, vocabulary, model)
    if verbose:
        print('')

    adata.uns[f'{vkey}_graph'] = vgraph.graph
    adata.uns[f'{vkey}_graph_neg'] = vgraph.graph_neg
    adata.obs[f'{vkey}_self_transition'] = vgraph.self_prob

    adata.layers[vkey] = np.zeros(adata.X.shape)

    return adata if copy else None


class VPT(DPT):
    def set_iroots(self, root=None):
        if (
            isinstance(root, str)
            and root in self._adata.obs.keys()
            and self._adata.obs[root].max() != 0
        ):
            #self.iroots = get_connectivities(self._adata).dot(self._adata.obs[root])
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
        rank_transform=False,
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
        'cell_fate' if groupby is None and 'cell_fate' in adata.obs.keys()
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
            scv.tl.terminal_states(adata, vkey, groupby, groups)
            root_key, end_key = 'root_nodes', 'end_points'
        cell_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        data = adata.copy() if cell_subset is None else adata[cell_subset].copy()
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
        pseudotime[cell_subset] = vpt.pseudotime
        adata.obs[f'{vkey}_pseudotime'] = np.array(pseudotime, dtype=np.float64)

        if save_diffmap:
            diffmap = np.empty(shape=(adata.n_obs, n_dcs))
            diffmap[:] = np.nan
            diffmap[cell_subset] = vpt.eigen_basis
            adata.obsm[f'X_diffmap_{groups}'] = diffmap

    return vpt if return_model else None

def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles='xy',
        scale_units='xy',
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor

def compute_velocity_on_grid(
        X_emb,
        V_emb,
        density=None,
        smooth=None,
        n_neighbors=None,
        min_mass=None,
        autoscale=True,
        adjust_for_stream=False,
        cutoff_perc=None,
        return_mesh=False,
):
    # remove invalid cells
    idx_valid = np.isfinite(X_emb.sum(1) + V_emb.sum(1))
    X_emb = X_emb[idx_valid]
    V_emb = V_emb[idx_valid]

    # prepare grid
    n_obs, n_dim = X_emb.shape
    density = 1 if density is None else density
    smooth = 0.5 if smooth is None else smooth

    grs = []
    for dim_i in range(n_dim):
        m, M = np.min(X_emb[:, dim_i]), np.max(X_emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(50 * density))
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    # estimate grid velocities
    if n_neighbors is None:
        n_neighbors = int(n_obs / 50)
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(X_emb)
    dists, neighs = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    weight = ss.norm.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    V_grid = (V_emb[neighs] * weight[:, :, None]).sum(1)
    V_grid /= np.maximum(1, p_mass)[:, None]
    if min_mass is None:
        min_mass = 1

    if adjust_for_stream:
        X_grid = np.stack([np.unique(X_grid[:, 0]), np.unique(X_grid[:, 1])])
        ns = int(np.sqrt(len(V_grid[:, 0])))
        V_grid = V_grid.T.reshape(2, ns, ns)

        mass = np.sqrt((V_grid ** 2).sum(0))
        min_mass = 10 ** (min_mass - 6)  # default min_mass = 1e-5
        min_mass = np.clip(min_mass, None, np.max(mass) * 0.9)
        cutoff = mass.reshape(V_grid[0].shape) < min_mass

        if cutoff_perc is None:
            cutoff_perc = 5
        length = np.sum(np.mean(np.abs(V_emb[neighs]), axis=1), axis=1).T
        length = length.reshape(ns, ns)
        cutoff |= length < np.percentile(length, cutoff_perc)

        V_grid[0][cutoff] = np.nan
    elif min_mass:
        min_mass *= np.percentile(p_mass, 99) / 100
        X_grid, V_grid = X_grid[p_mass > min_mass], V_grid[p_mass > min_mass]

        if autoscale:
            V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    if return_mesh:
        return X_grid, meshes_tuple, V_grid

    return X_grid, V_grid

def plot_pseudotime(
        adata,
        pfkey='pseudotime',
        rank_transform=False,
        use_ends=False,
        fill=True,
        levels=10,
        basis=None,
        vkey='velocity',
        density=None,
        smooth=None,
        pf_smooth=None,
        min_mass=None,
        arrow_size=None,
        arrow_length=None,
        arrow_color=None,
        scale=None,
        autoscale=True,
        n_neighbors=None,
        recompute=None,
        X=None,
        V=None,
        X_grid=None,
        V_grid=None,
        PF_grid=None,
        color=None,
        layer=None,
        color_map=None,
        colorbar=True,
        palette=None,
        size=None,
        alpha=0.5,
        offset=1,
        vmin=None,
        vmax=None,
        perc=None,
        sort_order=True,
        groups=None,
        components=None,
        projection='2d',
        legend_loc='none',
        legend_fontsize=None,
        legend_fontweight=None,
        xlabel=None,
        ylabel=None,
        title=None,
        fontsize=None,
        figsize=None,
        dpi=None,
        frameon=None,
        show=None,
        save=None,
        ax=None,
        ncols=None,
        **kwargs,
):
    if pfkey not in adata.obs:
        velocity_pseudotime(
            adata,
            vkey=vkey,
            groups=groups,
            rank_transform=rank_transform,
            use_velocity_graph=True,
            use_ends=use_ends,
        )
        adata.obs[pfkey] = adata.obs[f'{vkey}_pseudotime']

    smooth = 0.5 if smooth is None else smooth
    pf_smooth = smooth if pf_smooth is None else pf_smooth

    basis = scvu.default_basis(adata, **kwargs) \
            if basis is None \
            else scvu.get_basis(adata, basis)
    if vkey == 'all':
        lkeys = list(adata.layers.keys())
        vkey = [key for key in lkeys if 'velocity' in key and '_u' not in key]
    color, color_map = kwargs.pop('c', color), kwargs.pop('cmap', color_map)
    colors = scvu.make_unique_list(color, allow_array=True)
    layers, vkeys = (scvu.make_unique_list(layer),
                     scvu.make_unique_list(vkey))

    if V is None:
        for key in vkeys:
            if recompute or scvu.velocity_embedding_changed(
                    adata, basis=basis, vkey=key
            ):
                scv.pl.velocity_embedding(adata, basis=basis, vkey=key)

    color, layer, vkey = colors[0], layers[0], vkeys[0]
    color = scvu.default_color(adata) if color is None else color

    _adata = (
        adata[scvu.groups_to_bool(adata, groups, groupby=color)]
        if groups is not None and color in adata.obs.keys()
        else adata
    )
    comps, obsm = scvu.get_components(components, basis), _adata.obsm
    X_emb = np.array(obsm[f'X_{basis}'][:, comps]) \
            if X is None else X[:, :2]
    V_emb = np.array(obsm[f'{vkey}_{basis}'][:, comps]) \
            if V is None else V[:, :2]
    if X_grid is None or V_grid is None:
        X_grid, V_grid = compute_velocity_on_grid(
            X_emb=X_emb,
            V_emb=V_emb,
            density=density,
            autoscale=autoscale,
            smooth=smooth,
            n_neighbors=n_neighbors,
            min_mass=min_mass,
        )

    if vmin is None:
        vmin = adata.obs[pfkey].min()

    contour_kwargs = {
        'levels': levels,
        'vmin': vmin,
        'vmax': vmax,
        'alpha': alpha,
        'legend_fontsize': legend_fontsize,
        'legend_fontweight': legend_fontweight,
        'palette': palette,
        'cmap': color_map,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'colorbar': colorbar,
        'dpi': dpi,
    }

    ax, show = scvu.get_ax(ax, show, figsize, dpi)
    hl, hw, hal = scvu.default_arrow(arrow_size)
    if arrow_length is not None:
        scale = 1 / arrow_length
    if scale is None:
        scale = 1
    if arrow_color is None:
        arrow_color = 'grey'
    quiver_kwargs = {'angles': 'xy', 'scale_units': 'xy', 'edgecolors': 'k'}
    quiver_kwargs.update({'scale': scale, 'width': 0.001, 'headlength': hl / 2})
    quiver_kwargs.update({'headwidth': hw / 2, 'headaxislength': hal / 2})
    quiver_kwargs.update({'color': arrow_color, 'linewidth': 0.2, 'zorder': 3})

    for arg in list(kwargs):
        if arg in quiver_kwargs:
            quiver_kwargs.update({arg: kwargs[arg]})
        else:
            scatter_kwargs.update({arg: kwargs[arg]})

    ax.quiver(
        X_grid[:, 0], X_grid[:, 1], V_grid[:, 0], V_grid[:, 1], **quiver_kwargs
    )

    PF_emb = np.array(adata.obs[pfkey]).reshape(-1, 1)
    if offset is not None:
        PF_emb += offset

    if PF_grid is None:
        _, mesh, PF_grid = compute_velocity_on_grid(
            X_emb=X_emb,
            V_emb=PF_emb,
            density=density,
            autoscale=False,
            smooth=pf_smooth,
            n_neighbors=n_neighbors,
            min_mass=0.,
            return_mesh=True,
        )
        PF_grid = PF_grid.reshape(mesh[0].shape)

    if fill:
        contour_fn = ax.contourf
    else:
        contour_fn = ax.contour

    contour = contour_fn(mesh[0], mesh[1], PF_grid, zorder=1,
                         **contour_kwargs)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #cbar = plt.colorbar(contour)
    #cbar.ax.set_ylabel(pfkey)

    scvu.savefig_or_show(dpi=dpi, save=save, show=show)
    if show is False:
        return ax


def shortest_path(
        adata,
        source_idx,
        target_idx,
        vkey='velocity',
):
    if np.min((get_neighs(adata, 'distances') > 0).sum(1).A1) == 0:
        raise ValueError(
            'Your neighbor graph seems to be corrupted. '
            'Consider recomputing via scanpy.pp.neighbors.'
        )

    if f'{vkey}_graph' not in adata.uns:
        raise ValueError(
            'Must run velocity_graph() first.'
        )

    T = adata.uns[f'{vkey}_graph'] - adata.uns[f'{vkey}_graph_neg']

    import networkx as nx

    G = nx.convert_matrix.from_scipy_sparse_matrix(T)

    path = nx.algorithms.shortest_paths.generic.shortest_path(
        G, source=source_idx, target=target_idx,
    )

    return path


def plot_path(
        adata,
        path=None,
        source_idx=None,
        target_idx=None,
        basis='umap',
        vkey='velocity',
        ax=None,
        color='white',
        cmap=None,
        size=15,
        edgecolor='black',
        linecolor='#888888',
        linewidth=0.001,
):
    if path is None and (source_idx is None or target_idx is None):
        raise ValueError(
            'Must provide path indices or source and target indices.'
        )

    if path is None:
        path = shortest_path(adata, source_idx, target_idx, vkey=vkey)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    if f'X_{basis}' not in adata.obsm:
        raise ValueError(
            f'Basis {basis} not found in AnnData.'
        )

    basis_x = np.array(adata.obsm[f'X_{basis}'][path, 0]).ravel()
    basis_y = np.array(adata.obsm[f'X_{basis}'][path, 1]).ravel()

    for idx, (x, y) in enumerate(zip(basis_x, basis_y)):
        if idx < len(basis_x) - 1:
            dx, dy = basis_x[idx + 1] - x, basis_y[idx + 1] - y
            ax.arrow(x, y, dx, dy, width=linewidth, head_width=0,
                     length_includes_head=True,
                     color=linecolor, zorder=5)

    ax.scatter(basis_x, basis_y,
               s=size, c=color, cmap=cmap,
               edgecolors=edgecolor, linewidths=0.5, zorder=10)

    return ax

def tool_onehot_msa(
        adata,
        reference=None,
        key='onehot',
        seq_key='seq',
        backend='mafft',
        dirname='target/evolocity_alignments',
        n_threads=1,
        copy=False,
):
    # Write unaligned fasta.

    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    seqs = [
        SeqRecord(Seq(seq), id='seq{}'.format(idx), description='')
        for idx, seq in enumerate(adata.obs[seq_key])
    ]

    if dirname.endswith('/'):
        dirname = dirname.rstrip('/')
    mkdir_p(dirname)
    ifname = dirname + '/unaligned.fasta'
    SeqIO.write(seqs, ifname, 'fasta')

    # Align fasta.

    if backend == 'mafft':
        command = (
            'mafft ' +
            '--thread {} '.format(n_threads) +
            '--auto --treeout --inputorder ' +
            ifname
        ).split()
    else:
        raise ValueError('Unsupported backend: {}'.format(backend))

    import subprocess
    ofname = dirname + '/aligned.fasta'
    with open(ofname, 'w') as ofile, \
         open(dirname + '/' + backend + '.log', 'w') as olog:
        subprocess.run(command, stdout=ofile, stderr=olog)

    # Read alignment and turn to one-hot encoding.

    from Bio import AlignIO
    with open(ofname) as f:
        alignment = AlignIO.read(f, 'fasta')

    n_seqs = len(alignment)
    assert(n_seqs == adata.X.shape[0])
    if reference is not None:
        ref_aseq = str(alignment[reference].seq)
        n_residues = len(ref_aseq.replace('-', ''))
    else:
        n_residues = len(alignment[0].seq)
    align_matrix = np.zeros((n_seqs, n_residues))

    vocabulary = {}

    for i, record in enumerate(alignment):
        assert(record.id == 'seq{}'.format(i))
        aseq = str(record.seq)
        j = 0
        for char_idx, char in enumerate(aseq):
            if reference is not None and ref_aseq[char_idx] == '-':
                continue
            if char not in vocabulary:
                vocabulary[char] = len(vocabulary)
            align_matrix[i, j] = vocabulary[char]
            j += 1

    keys = sorted([ vocabulary[key] for key in vocabulary ])
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(
        categories=[ keys ] * align_matrix.shape[1],
        sparse=False,
    )
    X_onehot = enc.fit_transform(align_matrix)
    assert(X_onehot.shape[1] == len(keys) * n_residues)

    lookup = { vocabulary[key]: key for key in vocabulary }

    adata.obsm[f'X_{key}'] = X_onehot
    adata.obs[f'seqs_msa'] = [ str(record.seq) for record in alignment ]
    adata.uns[f'{key}_vocabulary'] = lookup
    adata.uns[f'{key}_shape'] = [ n_residues, len(lookup) ]

    return adata if copy else None

def tool_residue_scores(
        adata,
        basis='onehot',
        scale=1.,
        key='residue_scores',
        copy=False,
):
    if f'X_{basis}' not in adata.obsm:
        raise ValueError(f'Could not find basis "{basis}", '
                         'consider running onehot_msa() first.')

    scv.tl.velocity_embedding(
        adata,
        basis=basis,
        scale=scale,
        autoscale=False,
    )

    onehot_velo = np.array(adata.obsm[f'velocity_{basis}'])

    adata.uns[key] = onehot_velo.sum(0).reshape(
        tuple(adata.uns[f'{basis}_shape'])
    )

    return adata if copy else None

def plot_residue_scores(
        adata,
        percentile_keep=75,
        basis='onehot',
        key='residue_scores',
        cmap='RdBu',
        save=None,
):
    scores = AnnData(adata.uns[key])

    vocab = adata.uns[f'{basis}_vocabulary']
    scores.var_names = [
        vocab[key] for key in sorted(vocab.keys())
    ]

    positions = [ str(x) for x in range(scores.X.shape[0]) ]
    scores.obs['position'] = positions

    if percentile_keep > 0:
        score_sum = np.abs(scores.X).sum(1)
        cutoff = np.percentile(score_sum, percentile_keep)
        scores = scores[score_sum >= cutoff]

    end = max(abs(np.min(scores.X)), np.max(scores.X)) # Zero-centered colors.
    scores.X /= end # Scale within -1 and 1, inclusive.

    plt.figure(figsize=(
        max(scores.X.shape[1] // 2, 5),
        max(scores.X.shape[0] // 20, 5)
    ))
    sns.heatmap(
        scores.X,
        xticklabels=scores.var_names,
        yticklabels=scores.obs['position'],
        cmap=cmap,
        vmin=-1.,
        vmax=1.,
    )

    if save is not None:
        plt.savefig('figures/evolocity_' + save)
        plt.close()
    else:
        return ax

def plot_residue_categories(
        adata,
        positions=None,
        n_plot=5,
        namespace='residue_categories',
        reference=None
):
    if reference is not None:
        seq_ref = adata.obs['seq'][reference]
        seq_ref_msa = adata.obs['seqs_msa'][reference]
        pos2msa, ref_idx = {}, 0
        for idx, ch in enumerate(seq_ref_msa):
            if ch == '-':
                continue
            assert(ch == seq_ref[ref_idx])
            pos2msa[ref_idx] = idx
            ref_idx += 1

    if positions is None:
        scores = adata.uns['residue_scores']
        pos_seen = set()
        while len(pos_seen) < n_plot:
            min_idx = np.unravel_index(np.argmin(scores), scores.shape)
            scores[min_idx] = float('inf')
            aa = adata.uns['onehot_vocabulary'][min_idx[1]]
            pos = min_idx[0]
            if pos in pos_seen:
                continue
            pos_seen.add(pos)
            tprint('Lowest score {}: {}{}'.format(len(pos_seen), aa, pos + 1))
        positions = sorted(pos_seen)

    for pos in positions:
        adata.obs[f'pos{pos}'] = [
            seq[pos] if reference is None else seq[pos2msa[pos]]
            for seq in adata.obs['seqs_msa']
        ]
        sc.pl.umap(adata, color=f'pos{pos}', save=f'_{namespace}_pos{pos}.png',
                   edges=True,)

def load_uniref(ds_fname, map_fname=None):
    if ds_fname.endswith('.fasta') or ds_fname.endswith('.fa'):
        uniref_seqs = set([
            str(record.seq) for record in SeqIO.parse(ds_fname, 'fasta')
        ])
    else:
        raise ValueError(f'Invalid extension for file "{ds_fname}"')

    if map_fname is None:
        return uniref_seqs

    cluster_map = {
        record.id: str(record.seq)
        for record in SeqIO.parse(ds_fname, 'fasta')
    }
    with open(map_fname) as f:
        for line in f:
            fields = line.rstrip().split()
            cluster_map[fields[0]] = fields[-1]

    return uniref_seqs, cluster_map

def check_uniref50(
        adata,
        key='uniref50',
        verbose=True,
        id_key='gene_id',
        ds_fname='data/uniref/uniref50_2018_03.fasta',
):
    uniref_seqs = load_uniref(ds_fname)
    is_uniref = [ seq in uniref_seqs for seq in adata.obs['seq'] ]

    if verbose:
        tprint('UniRef50 seqs:')
        for gene_id in adata[is_uniref].obs[id_key]:
            tprint(f'\t{gene_id}')

    adata.obs[key] = is_uniref

def training_distances(
        seqs,
        namespace='',
        key='homology',
        accession_key='accession',
        dataset='uniref',
        ds_fname='data/uniref/uniref50_2018_03.fasta',
        map_fname='data/uniref/uniref50_2018_03_mapping.txt',
        exact_search=True,
):
    dirname = 'target/training_seqs'
    if namespace:
        dirname += f'/{namespace}'
    mkdir_p(dirname)

    # Get set of closest training sequences.

    fname = dirname + f'/training_{dataset}.txt'
    if os.path.isfile(fname):
        with open(fname) as f:
            training_seqs = f.read().rstrip().split()
    else:
        dataset_seqs, cluster_map = load_uniref(ds_fname, map_fname)

        training_seqs = (set([
            seq for seq in seqs if str(seq) in dataset_seqs
        ]) | set([
            cluster_map[cluster_map[meta[accession_key]]]
            for seq in seqs for meta in seqs[seq]
            if accession_key in meta and meta[accession_key] in cluster_map
        ]))

        if namespace:
            with open(fname, 'w') as of:
                for seq in sorted(training_seqs):
                    of.write(str(seq) + '\n')

    # Compute distance to closest training sequence.

    if exact_search:
        for seq in seqs:
            ratio = fuzzyproc.extractOne(str(seq), training_seqs)[1]
            for meta in seqs[seq]:
                meta[key] = float(ratio)

    else:
        from datasketch import MinHash, MinHashLSHForest
        from nltk import ngrams

        # Index training sequences.
        lsh = MinHashLSHForest(num_perm=128)
        for seq in training_seqs:
            seq = str(seq)
            minhash = MinHash(num_perm=128)
            for d in ngrams(seq, 5):
                minhash.update(''.join(d).encode('utf-8'))
            lsh.add(seq, minhash)
        lsh.index()

        # Query data structure for (approximately) closest seqs.
        for seq in seqs:
            minhash = MinHash(num_perm=128)
            for d in ngrams(seq, 5):
                minhash.update(''.join(d).encode('utf-8'))
            result = lsh.query(minhash, 1)[0]
            ratio = fuzz.ratio(str(seq), result)
            for meta in seqs[seq]:
                meta[key] = float(ratio)

    return seqs

def plot_ancestral(
        df,
        meta_key,
        name_key='name',
        score_key='score',
        homology_key='homology',
        namespace='ancestral',
):
    if meta_key not in df:
        raise ValueError('Metadata key not found.')
    if name_key not in df:
        raise ValueError('Ancestral names key not found.')
    if score_key not in df:
        raise ValueError('Likelihood scores key not found.')

    for name in set(df[name_key]):
        df_name = df[df[name_key] == name]

        plt.figure()
        sns.boxplot(
            data=df_name, x=meta_key, y=score_key,
        )
        plt.axhline(y=0, c='#CCCCCC', linestyle='dashed')
        name_sanitized = name.replace('/', '-')
        plt.savefig(f'figures/{namespace}_ancestral_{name_sanitized}.svg',
                    dpi=500)
        plt.close()

        if homology_key in df_name:
            r, p = ss.spearmanr(df_name[score_key].values,
                                df_name[homology_key].values,
                                nan_policy='omit')
            tprint('{} corr with ancestral: Spearman r = {}, P = {}'.format(
                name, r, p
            ))

        for meta in set(df_name[meta_key]):
            score_dist = df_name[df_name[meta_key] == meta].score.values
            tprint('{} and {}: {}% percentile'.format(
                name, meta,
                ss.percentileofscore(score_dist, 0)
            ))
