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

    seqs = { seq_of_interest: [ {} ] }
    X_cat, lengths = featurize_seqs(seqs, vocabulary)

    y_pred = model.predict(X_cat, lengths)
    assert(y_pred.shape[0] == len(seq_of_interest) + 2)

    return y_pred

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

def likelihood_muts(seq1, seq2, vocabulary, model,
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
        seq1, seq2, vocabulary, model,
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


    def compute_likelihoods(self, vocabulary, model):
        if self.verbose:
            iterator = tqdm(self.seqs)
        else:
            iterator = self.seqs

        for seq in iterator:
            y_pred = predict_sequence_prob(
                seq, vocabulary, model, verbose=self.verbose
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
                raise ValueError('Invalid score {}'.format(self.score))


    def compute_gradients(self, vocabulary, model):
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
                    vocabulary, model,
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

    valid_scores = { 'other' }
    if score not in valid_scores:
        raise ValueError('Score must be one of {}'
                         .format(', '.join(valid_scores)))

    if mkey not in adata.uns or model_name != adata.uns[mkey].name_:
        velocity_model(
            adata,
            model_name=model_name,
            mkey=mkey,
        )
    model = velocity_model[mkey]
    vocabulary = model.vocabulary_

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
