from .. import logging as logg
from ..preprocessing.neighbors import (
    compute_connectivities_umap,
    get_neighs,
    neighbors,
    verify_neighbors,
)
from ..preprocessing.utils import sum_var
from .utils import scale
from .velocity_model import velocity_model

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
from scipy.sparse import coo_matrix
import numpy as np
from tqdm import tqdm

# Choices of scoring functions.
SCORE_CHOICES = {
    'lm',
    'unit',
    'random',
    'edgerand',
}

# Choices of substitution matrices.
SUBMAT_CHOICES = {
    'blosum62',
    'jtt',
    'wag',
}

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

def predict_sequence_prob(seq_of_interest, vocabulary, model,
                          verbose=False):
    if 'esm' in model.name_:
        from .fb_semantics import predict_sequence_prob_fb
        return predict_sequence_prob_fb(
            seq_of_interest, model.alphabet_, model.model_,
            model.repr_layers_, verbose=verbose,
        )
    elif model.name_ == 'tape':
        from .tape_semantics import predict_sequence_prob_tape
        return predict_sequence_prob_tape(
            seq_of_interest, model
        )
    else:
        raise ValueError('Invalid model name {}'.format(model.name_))

def likelihood_compare(seq1, seq2, vocabulary, model,
                       pos1=None, pos2=None, seq_cache={}, verbose=False):
    likelihoods = []

    for seq_pred, positions in zip([ seq1, seq2 ], [ pos1, pos2 ]):
        if positions is None:
            positions = range(len(seq_pred))

        if seq_pred in seq_cache:
            seq_probs = seq_cache[seq_pred][list(positions)]

        else:
            y_pred = predict_sequence_prob(
                seq_pred, vocabulary, model, verbose=verbose
            )
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

def align_seqs(seq1, seq2):
    # Align, prefer matches to gaps.
    return pairwise2.align.globalms(
        seq1, seq2, 5, -4, -4, -.1, one_alignment_only=True
    )[0]

def likelihood_muts(
        seq1, seq2, vocabulary, model,
        seq_cache={}, verbose=False, natural_aas=None,
):
    a_seq1, a_seq2, _, _, _ = align_seqs(seq1, seq2)

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
            if natural_aas is not None and \
               (ch.upper() not in natural_aas or \
                other_seq[a_idx].upper() not in natural_aas):
                continue
            if other_seq[a_idx] != ch:
                substitutions.append(orig_idx)
            orig_idx += 1

    return likelihood_compare(
        seq1, seq2, vocabulary, model,
        pos1=sub1, pos2=sub2, seq_cache=seq_cache, verbose=verbose,
    )

def likelihood_submat(
        seq1, seq2, matrix, vocabulary, model,
        seq_cache={}, verbose=False, natural_aas=None,
):
    a_seq1, a_seq2, _, _, _ = align_seqs(seq1, seq2)

    scores = []
    for ch1, ch2 in zip(a_seq1, a_seq2):
        if ch1 == ch2:
            continue
        if (ch1, ch2) in matrix:
            scores.append(matrix[(ch1, ch2)])
        elif (ch2, ch1) in matrix:
            scores.append(matrix[(ch2, ch1)])

    return np.mean(scores)

def likelihood_blosum62(
        seq1, seq2, vocabulary, model,
        seq_cache={}, verbose=False, natural_aas=None,
):
    from Bio.SubsMat import MatrixInfo as matlist
    matrix = matlist.blosum62
    return likelihood_submat(
        seq1, seq2, matrix, vocabulary, model,
        seq_cache, verbose, natural_aas,
    )

def likelihood_jtt(
        seq1, seq2, vocabulary, model,
        seq_cache={}, verbose=False, natural_aas=None,
):
    from Bio.SubsMat import read_text_matrix
    with open('data/substitution_matrices/JTT.txt') as f:
        matrix = read_text_matrix(f)
    return likelihood_submat(
        seq1, seq2, matrix, vocabulary, model,
        seq_cache, verbose, natural_aas,
    )

def likelihood_wag(
        seq1, seq2, vocabulary, model,
        seq_cache={}, verbose=False, natural_aas=None,
):
    from Bio.SubsMat import read_text_matrix
    with open('data/substitution_matrices/WAG.txt') as f:
        matrix = read_text_matrix(f)
    return likelihood_submat(
        seq1, seq2, matrix, vocabulary, model,
        seq_cache, verbose, natural_aas,
    )

def likelihood_unit(*args, **kwargs):
    # For control experiment, return unit velocities.
    return 1.
    
def likelihood_random(*args, **kwargs):
    # For control experiment, return Gaussian noise.
    return np.random.normal(loc=5, scale=10)
    
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
            score='lm',
            vkey='velocity',
            n_recurse_neighbors=None,
            random_neighbors_at_max=None,
            mode_neighbors='distances',
            include_set='natural_aas',
            verbose=False,
    ):
        self.adata = adata

        self.seqs = seqs
        self.seq_probs = {}

        self.score = score

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if mode_neighbors == 'connectivities':
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if include_set == 'natural_aas':
            self.include_set = set([
                'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
            ])
        else:
            self.include_set = None

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


    def compute_likelihoods(self, vocabulary, model):
        if self.verbose:
            iterator = tqdm(self.seqs)
        else:
            iterator = self.seqs

        if self.score != 'lm' and self.score != 'edgerand':
            return

        for seq in iterator:
            y_pred = predict_sequence_prob(
                seq, vocabulary, model, verbose=self.verbose
            )

            if self.score == 'lm' or self.score == 'edgerand':
                self.seq_probs[seq] = np.array([
                    y_pred[i + 1, (
                        vocabulary[seq[i]]
                        if seq[i] in vocabulary else
                        model.unk_idx_
                    )] for i in range(len(seq))
                ])
            else:
                raise ValueError('Invalid score {}'.format(self.score))


    def compute_gradients(self, vocabulary, model):
        n_obs = self.adata.X.shape[0]
        vals, rows, cols, uncertainties = [], [], [], []

        if self.verbose:
            iterator = tqdm(range(n_obs))
        else:
            iterator = range(n_obs)

        # Iterate over edges and compute velocity score for each edge.
        for i in iterator:
            neighs_idx = get_iterative_indices(
                self.indices, i, self.n_recurse_neighbors, self.max_neighs
            )

            if self.score == 'lm':
                score_fn = likelihood_muts
            elif self.score == 'blosum62':
                score_fn = likelihood_blosum62
            elif self.score == 'jtt':
                score_fn = likelihood_jtt
            elif self.score == 'wag':
                score_fn = likelihood_wag
            elif self.score == 'unit':
                score_fn = likelihood_unit
            elif self.score == 'random':
                score_fn = likelihood_random
            elif self.score == 'edgerand':
                # Compute velocity with random edges.
                score_fn = likelihood_muts
                neighs_idx = np.random.choice(
                    len(self.seqs),
                    size=len(neighs_idx),
                    replace=False
                )
            else:
                raise ValueError('Invalid score {}'.format(self.score))

            val = np.array([
                score_fn(
                    self.seqs[i], self.seqs[j],
                    vocabulary, model,
                    seq_cache=self.seq_probs, verbose=self.verbose,
                    natural_aas=self.include_set,
                ) for j in neighs_idx
            ])

            vals.extend(val)
            rows.extend(np.ones(len(neighs_idx)) * i)
            cols.extend(neighs_idx)

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

def velocity_graph(
        adata,
        model_name='esm1b',
        mkey='model',
        score='lm',
        seqs=None,
        vkey='velocity',
        n_recurse_neighbors=0,
        random_neighbors_at_max=None,
        mode_neighbors='distances',
        include_set=None,
        copy=False,
        verbose=True,
):
    """Computes velocity scores at each edge in the graph.

    At each edge connecting two sequences :math:`(x^{(a)}, x^{(b)})`,
    computes a score

    .. math::
        v_{ab} = \\frac{1}{|\\mathcal{M}|} \\sum_{i \in \\mathcal{M}}
        \\left[ \\log p\\left( x_i^{(b)} | x^{(a)} \\right) -
        \\log p\\left( x_i^{(a)} | x^{(b)} \\right) \\right]

    where :math:`\\mathcal{M} = \\left\\{ i : x_i^{(a)} \\neq x_i^{(b)} \\right\\}`
    is the set of positions at which the amino acid residues disagree.

    Arguments
    ---------
    adata: :class:`~anndata.Anndata`
        Annoated data matrix.
    model_name: `str` (default: `'esm1b'`)
        Language model used to compute likelihoods.
    mkey: `str` (default: `'model'`)
        Name at which language model is stored.
    score: `str` (default: `'lm'`)
        Type of velocity score.
    seqs: `list` (default: `'None'`)
        List of sequences; defaults to those in `adata.obs['seq']`.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    n_recurse_neighbors: `int` (default: `0`)
        Number of recursions for neighbors search.
    random_neighbors_at_max: `int` or `None` (default: `None`)
        If number of iterative neighbors for an individual node is higher than this
        threshold, a random selection of such are chosen as reference neighbors.
    mode_neighbors: `str` (default: `'distances'`)
        Determines the type of KNN graph used. Options are 'distances' or
        'connectivities'. The latter yields a symmetric graph.
    include_set: `set` (default: `None`)
        Set of characters to explicitly include.
    verbose: `bool` (default: `True`)
        Print logging output.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.

    Returns
    -------
    Returns or updates `adata` with the attributes
    model: `.uns`
        language model
    velocity_graph: `.uns`
        sparse matrix with transition probabilities
    """

    adata = adata.copy() if copy else adata
    verify_neighbors(adata)

    if seqs is None:
        seqs = adata.obs['seq']
    if adata.X.shape[0] != len(seqs):
        raise ValueError('Number of sequences should correspond to '
                         'number of observations.')

    valid_scores = SCORE_CHOICES | SUBMAT_CHOICES
    if score not in valid_scores:
        raise ValueError('Score must be one of {}'
                         .format(', '.join(valid_scores)))

    if mkey not in adata.uns or model_name != adata.uns[mkey].name_:
        velocity_model(
            adata,
            model_name=model_name,
            mkey=mkey,
        )
    model = adata.uns[mkey]
    vocabulary = model.vocabulary_

    vgraph = VelocityGraph(
        adata,
        seqs,
        score=score,
        vkey=vkey,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        mode_neighbors=mode_neighbors,
        include_set=include_set,
        verbose=verbose,
    )

    if verbose:
        logg.msg('Computing likelihoods...')
    vgraph.compute_likelihoods(vocabulary, model)
    if verbose:
        print('')

    if verbose:
        logg.msg('Computing velocity graph...')
    vgraph.compute_gradients(vocabulary, model)
    if verbose:
        print('')

    adata.uns[f'{vkey}_graph'] = vgraph.graph
    adata.uns[f'{vkey}_graph_neg'] = vgraph.graph_neg
    adata.obs[f'{vkey}_self_transition'] = vgraph.self_prob

    adata.layers[vkey] = np.zeros(adata.X.shape)

    return adata if copy else None
