from .utils import mkdir_p

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

def onehot_msa(
        adata,
        reference=None,
        seq_id_fields=None,
        key='onehot',
        seq_key='seq',
        backend='mafft',
        dirname='target/evolocity_alignments',
        n_threads=1,
        copy=False,
):
    """Aligns and one-hot-encodes sequences.

    By default, uses the MAFFT aligner (https://mafft.cbrc.jp/alignment/software/),
    which can be installed via conda using

    .. code:: bash

        conda install -c bioconda mafft

    Arguments
    ---------
    adata: :class:`~anndata.Anndata`
        Annoated data matrix.
    reference: `int` (default: None)
        Index corresponding to a sequence in `adata` to be used as the main
        reference sequence for the alignment.
    seq_id_fields: `list` (default: None)
        List of fields in `adata.obs` to store in FASTA IDs.
    key: `str` (default: `'onehot'`)
        Name at which the embedding is stored.
    seq_key: `str` (default: `'seq'`)
        Name of sequences in `.obs`.
    backend: `str` (default: `None` )
        Sequence alignment tool.
    dirname: `str` (default: `'target/evolocity_alignments'`)
        Directory under which to place alignment files.
    n_threads: `int` (default: 1)
        Number of threads for sequence alignment.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.

    Returns
    -------
    Returns or updates `adata` with the attributes
    X_onehot: `.obsm`
        one-hot embeddings
    seqs_msa: `.obs`
        aligned sequences
    """

    # Write unaligned fasta.

    seqs = []
    for idx, seq in enumerate(adata.obs[seq_key]):
        seq_id = f'seq{idx}'
        if seq_id_fields is not None:
            for field in seq_id_fields:
                seq_id += f'_{field}{adata.obs[field][idx]}'
        seqs.append(SeqRecord(Seq(seq), id=seq_id, description=''))

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
        assert(record.id.startswith('seq{}'.format(i)))
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
    """Score mutations by associated evolocity.

    Requires running ``evo.tl.onehot_msa`` first.

    Arguments
    ---------
    adata: :class:`~anndata.Anndata`
        Annoated data matrix.
    basis: `str` (default: `'onehot'`)
        Name of one-hot embedding
    scale: `float` (default: `1.`)
        Scale parameter of gaussian kernel for transition matrix.
    key: `str` (default: `'residue_scores'`)
        Name at which to place scores.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.

    Returns
    -------
    Returns or updates `adata` with the attributes
    residue_scores: `.uns`
        per-residue velocity scores
    """
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
