from .utils import mkdir_p

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

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
