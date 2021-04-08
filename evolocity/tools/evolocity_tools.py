from ..preprocessing.moments import get_connectivities
from ..preprocessing.neighbors import (
    compute_connectivities_umap,
    get_n_neighs,
    get_neighs,
    neighbors,
    verify_neighbors,
)
from ..preprocessing.utils import sum_var
from .terminal_states import terminal_states
from .utils import groups_to_bool, scale, strings_to_categoricals

from Bio import pairwise2, SeqIO
from Bio.SubsMat import MatrixInfo as matlist
from scanpy.tools._dpt import DPT
from scipy.sparse import coo_matrix, issparse, spdiags, linalg
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

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

            score_fn = likelihood_muts

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

def onehot_msa(
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

def residue_scores(
        adata,
        basis='onehot',
        scale=1.,
        key='residue_scores',
        copy=False,
):
    if f'X_{basis}' not in adata.obsm:
        raise ValueError(f'Could not find basis "{basis}", '
                         'consider running onehot_msa() first.')

    from .velocity_embedding import velocity_embedding
    velocity_embedding(
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
