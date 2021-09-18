from .. import logging as logg
from .utils import mkdir_p

from anndata import AnnData
import math
import numpy as np
import os

def get_model(model_name):
    if model_name == 'esm1':
        from ..tools.fb_model import FBModel
        model = FBModel(
            'esm1_t34_670M_UR50S',
            repr_layer=[-1],
        )
    elif model_name == 'esm1b':
        from ..tools.fb_model import FBModel
        model = FBModel(
            'esm1b_t33_650M_UR50S',
            repr_layer=[-1],
        )
    elif model_name == 'esm1b-rand':
        from ..tools.fb_model import FBModel
        model = FBModel(
            'esm1b_t33_650M_UR50S',
            repr_layer=[-1],
            random_init=True,
        )
    elif model_name == 'tape':
        from ..tools.tape_model import TAPEModel
        model = TAPEModel(
            'bert-base',
        )
    else:
        raise ValueError('Invalid model {}'.format(model_name))

    return model

def embed_seqs(
    model,
    seqs,
    namespace,
    verbose=True,
):
    if 'esm' in model.name_:
        from ..tools.fb_semantics import embed_seqs_fb
        seqs_fb = sorted([ seq for seq in seqs ])
        embedded = embed_seqs_fb(
            model.model_, seqs_fb, model.repr_layers_, model.alphabet_,
            use_cache=False, verbose=verbose,
        )
        X_embed = np.array([
            embedded[seq][0]['embedding'] for seq in seqs_fb
        ])
    else:
        raise ValueError('Model {} not supported for sequence embedding'
                         .format(model.name_))

    sorted_seqs = sorted(seqs)
    for seq_idx, seq in enumerate(sorted_seqs):
        for meta in seqs[seq]:
            meta['embedding'] = X_embed[seq_idx]

    return seqs

def populate_embedding(
    model,
    seqs,
    namespace=None,
    use_cache=False,
    batch_size=3000,
    verbose=True,
):
    if namespace is None:
        namespace = 'protein'

    if use_cache:
        mkdir_p('target/{}/embedding'.format(namespace))
        embed_prefix = ('target/{}/embedding/{}_512'
                        .format(namespace, model.name_,))

    sorted_seqs = np.array([ str(s) for s in sorted(seqs.keys()) ])
    n_batches = math.ceil(len(sorted_seqs) / float(batch_size))
    for batchi in range(n_batches):
        if verbose:
            logg.info('Embedding sequence batch {} / {}'
                      .format(batchi + 1, n_batches))

        # Identify the batch.
        start = batchi * batch_size
        end = (batchi + 1) * batch_size
        sorted_seqs_batch = sorted_seqs[start:end]
        seqs_batch = { seq: seqs[seq] for seq in sorted_seqs_batch }

        # Load from cache if available.
        if use_cache:
            embed_fname = embed_prefix + '.{}.npy'.format(batchi)
            if os.path.exists(embed_fname):
                X_embed = np.load(embed_fname, allow_pickle=True)
                if X_embed.shape[0] == len(sorted_seqs_batch):
                    for seq_idx, seq in enumerate(sorted_seqs_batch):
                        for meta in seqs[seq]:
                            meta['embedding'] = X_embed[seq_idx]
                    continue

        # Embed the sequences.
        seqs_batch = embed_seqs(
            model,
            seqs_batch,
            namespace,
            verbose=verbose,
        )

        if use_cache:
            X_embed = []
        for seq in sorted_seqs_batch:
            for meta in seqs[seq]:
                meta['embedding'] = seqs_batch[seq][0]['embedding'].mean(0)
            if use_cache:
                X_embed.append(seqs[seq][0]['embedding'].ravel())
        del seqs_batch

        if use_cache:
            np.save(embed_fname, np.array(X_embed))

    return seqs

def seqs_to_anndata(seqs):
    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    obs['seq_len'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'])
        for key in meta:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            obs[key].append(Counter([
                meta[key] for meta in seqs[seq]
            ]).most_common(1)[0][0])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
        obs['seq_len'].append(len(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    return adata

def featurize_seqs(
    seqs,
    model_name='esm1b',
    mkey='model',
    embed_batch_size=3000,
    use_cache=False,
    cache_namespace='protein',
):
    """Embeds a list of sequences.

    Takes a list of sequences and returns an :class:`~anndata.Anndata`
    object with sequence embeddings in the `adata.X` matrix.

    Arguments
    ---------
    seqs: `list`
        List of protein sequences.
    model_name: `str` (default: `'esm1b'`)
        Language model used to compute likelihoods.
    mkey: `str` (default: `'model'`)
        Name at which language model is stored.
    embed_batch_size: `int` (default: `3000`)
        Batch size to embed sequences. Lower to fit into GPU memory.
    use_cache: `bool` (default: `False`)
        Cache embeddings to disk for faster future loading.
    cache_namespace: `str` (default: `'protein'`)
        Namespace at which to store cache.

    Returns
    -------
    Returns an :class:`~anndata.Anndata` object with the attributes
    `.X`
        Matrix where rows correspond to sequences and columns are
        language model embedding dimensions
    seq: `.obs`
        Sequences corresponding to rows in `adata.X`
    model: `.uns`
        language model
    """
    model = get_model(model_name)

    seqs = {
        str(seq): [ {} ] for seq in seqs
    }
    seqs = populate_embedding(
        model,
        seqs,
        namespace=cache_namespace,
        use_cache=use_cache,
        batch_size=embed_batch_size,
    )

    adata = seqs_to_anndata(seqs)

    adata.uns[f'featurize_{mkey}'] = model
    adata.uns[f'{mkey}'] = model

    return adata

def featurize_fasta(
    fname,
    model_name='esm1b',
    mkey='model',
    embed_batch_size=3000,
    use_cache=True,
    cache_namespace=None,
):
    """Embeds a FASTA file.

    Takes a FASTA file containing sequences and returns an
    :class:`~anndata.Anndata` object with sequence embeddings
    in the `adata.X` matrix.

    Assumes metadata is storred in FASTA record as `key=value`
    pairs that are separated by vertical bar "|" characters.

    Arguments
    ---------
    fname: `str`
        Path to FASTA file.
    model_name: `str` (default: `'esm1b'`)
        Language model used to compute likelihoods.
    mkey: `str` (default: `'model'`)
        Name at which language model is stored.
    embed_batch_size: `int` (default: `3000`)
        Batch size to embed sequences. Lower to fit into GPU memory.
    use_cache: `bool` (default: `False`)
        Cache embeddings to disk for faster future loading.
    cache_namespace: `str` (default: `'protein'`)
        Namespace at which to store cache.

    Returns
    -------
    Returns an :class:`~anndata.Anndata` object with the attributes
    `.X`
        Matrix where rows correspond to sequences and columns are
        language model embedding dimensions
    seq: `.obs`
        Sequences corresponding to rows in `adata.X`
    model: `.uns`
        language model
    """
    model = get_model(model_name)

    # Parse fasta.
    from Bio import SeqIO
    seqs = {}
    with open(fname, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            fields = record.id.split('|')
            meta = {
                field.split('=')[0]: field.split('=')[1]
                for field in fields
            }
            seq = str(record.seq)
            if seq not in seqs:
                seqs[seq] = []
            seqs[seq].append(meta)

    seqs = populate_embedding(
        model,
        seqs,
        namespace=cache_namespace,
        use_cache=use_cache,
        batch_size=embed_batch_size,
    )

    adata = seqs_to_anndata(seqs)

    adata.uns[mkey] = model
    adata.uns[f'featurize_{mkey}'] = model

    return adata
