from mutation import *
from evolocity_graph import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Cytochrome c sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='cyc',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--train-split', action='store_true',
                        help='Train model on portion of data')
    parser.add_argument('--test', action='store_true',
                        help='Test model')
    parser.add_argument('--ancestral', action='store_true',
                        help='Analyze ancestral sequences')
    parser.add_argument('--evolocity', action='store_true',
                        help='Analyze evolocity')
    args = parser.parse_args()
    return args

def load_taxonomy():
    tax_fnames = [
        'data/cyc/taxonomy_archaea.tab.gz',
        'data/cyc/taxonomy_bacteria.tab.gz',
        'data/cyc/taxonomy_eukaryota.tab.gz',
    ]

    import gzip

    taxonomy = {}

    for fname in tax_fnames:
        with gzip.open(fname) as f:
            header = f.readline().decode('utf-8').rstrip().split('\t')
            assert(header[0] == 'Taxon' and header[8] == 'Lineage')
            for line in f:
                fields = line.decode('utf-8').rstrip().split('\t')
                tax_id = fields[0]
                lineage = fields[8]
                taxonomy[tax_id] = lineage

    return taxonomy

def parse_meta(record, taxonomy):
    if 'GN=' in record:
        (_, accession, gene_id, name, species, species_id,
         gene_symbol, pe, sv) = record.split('|')
    else:
        (_, accession, gene_id, name, species, species_id,
         pe, sv) = record.split('|')
        gene_symbol = None

    tax_id = species_id[3:]
    lineage = taxonomy[tax_id]

    tax_group = None
    if 'Archaea' in lineage:
        tax_group = 'archaea'
    if 'Bacteria' in lineage:
        tax_group = 'bacteria'
    if 'Eukaryota' in lineage:
        tax_group = 'eukaryota'
    if 'Fungi' in lineage:
        tax_group = 'fungi'
    if 'Viridiplantae' in lineage:
        tax_group = 'viridiplantae'
    if 'Arthropoda' in lineage:
        tax_group = 'arthropoda'
    if 'Chordata' in lineage:
        tax_group = 'chordata'
    if 'Mammalia' in lineage:
        tax_group = 'mammalia'
    if 'Primate' in lineage:
        tax_group = 'primate'
    assert(tax_group is not None)

    return {
        'accession': accession,
        'gene_id': gene_id,
        'name': name,
        'species': species[3:],
        'tax_id': tax_id,
        'tax_group': tax_group,
        'lineage': lineage,
        'gene_symbol': gene_symbol[3:] if gene_symbol is not None else None,
        'pe': pe[3:],
        'sv': sv[3:],
    }

def process(fnames):
    taxonomy = load_taxonomy()

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 100 or len(record.seq) > 115:
                continue
            meta = parse_meta(record.id, taxonomy)
            if 'Eukaryota' not in meta['lineage']:
                continue
            if 'CYC6' in meta['gene_id'] or 'c6' in meta['name']:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            meta['seq_len'] = len(record.seq)
            seqs[record.seq].append(meta)

    seqs = training_distances(seqs, namespace=args.namespace)

    return seqs

def split_seqs(seqs, split_method='random'):
    raise NotImplementedError('split_seqs not implemented')

def setup(args):
    fnames = [ 'data/cyc/uniprot_cyc.fasta' ]

    import pickle
    cache_fname = 'target/ev_cache/cyc_seqs.pkl'
    try:
        with open(cache_fname, 'rb') as f:
            seqs = pickle.load(f)
    except:
        seqs = process(fnames)
        with open(cache_fname, 'wb') as of:
            pickle.dump(seqs, of)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

def plot_umap(adata, namespace='cyc'):
    sc.pl.umap(adata, color='tax_group', edges=True,
               save='_{}_taxonomy.png'.format(namespace))
    sc.pl.umap(adata, color='louvain', edges=True,
               save='_{}_louvain.png'.format(namespace))
    sc.pl.umap(adata, color='seq_len', edges=True,
               save='_{}_seqlen.png'.format(namespace))
    sc.pl.umap(adata, color='homology', edges=True,
               save='_{}_homology.png'.format(namespace))

def seqs_to_anndata(seqs):
    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
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
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    return adata

def cyc_ancestral(args, model, seqs, vocabulary, namespace='cyc'):
    path_fname = 'data/cyc/ancestral_cyc_curated_codeml.fa'
    nodes = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(path_fname, 'fasta')
    ]

    ######################################
    ## See how local likelihoods change ##
    ######################################

    tax_types = {
        'fungi',
        'chordata',
        'mammalia',
        'viridiplantae',
    }

    dist_data = []
    for idx, (name, seq) in enumerate(nodes):
        for uniprot_seq in seqs:
            tax_type = Counter([
                meta['tax_group'] for meta in seqs[uniprot_seq]
            ]).most_common(1)[0][0]
            if tax_type not in tax_types:
                continue
            if tax_type == 'fungi' and \
               ('all' not in name and 'fungi' not in name):
                continue
            if tax_type == 'chordata' and \
               ('all' not in name and 'animalia' not in name):
                continue
            if tax_type == 'mammalia' and \
               ('all' not in name and 'animalia' not in name):
                continue
            if tax_type == 'viridiplantae' and \
               ('all' not in name and 'plantae' not in name):
                continue
            score = likelihood_muts(seq, uniprot_seq,
                                    args, vocabulary, model,)
            homology = fuzz.ratio(seq, uniprot_seq)
            dist_data.append([ tax_type, name, score, homology ])

    df = pd.DataFrame(dist_data, columns=[
        'tax_type', 'name', 'score', 'homology'
    ])

    plot_ancestral(df, meta_key='tax_type', namespace=namespace)
    plot_ancestral(df, meta_key='name', name_key='tax_type', namespace=namespace)

def evo_cyc(args, model, seqs, vocabulary, namespace='cyc'):

    ######################################
    ## Visualize Cytochrome C landscape ##
    ######################################

    adata_cache = 'target/ev_cache/cyc_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        seqs = populate_embedding(args, model, seqs, vocabulary, use_cache=True)
        adata = seqs_to_anndata(seqs)
        adata.write(adata_cache)

    if 'homologous' in namespace:
        adata = adata[adata.obs['homology'] > 80.]

    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')

    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    sc.tl.umap(adata, min_dist=1.)
    plot_umap(adata, namespace=namespace)

    #####################################
    ## Compute evolocity and visualize ##
    #####################################

    cache_prefix = f'target/ev_cache/{namespace}_knn30'
    try:
        from scipy.sparse import load_npz
        adata.uns["velocity_graph"] = load_npz(
            '{}_vgraph.npz'.format(cache_prefix)
        )
        adata.uns["velocity_graph_neg"] = load_npz(
            '{}_vgraph_neg.npz'.format(cache_prefix)
        )
        adata.obs["velocity_self_transition"] = np.load(
            '{}_vself_transition.npy'.format(cache_prefix)
        )
        adata.layers["velocity"] = np.zeros(adata.X.shape)
    except:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        velocity_graph(adata, args, vocabulary, model,
                       n_recurse_neighbors=0,)
        from scipy.sparse import save_npz
        save_npz('{}_vgraph.npz'.format(cache_prefix),
                 adata.uns["velocity_graph"],)
        save_npz('{}_vgraph_neg.npz'.format(cache_prefix),
                 adata.uns["velocity_graph_neg"],)
        np.save('{}_vself_transition.npy'.format(cache_prefix),
                adata.obs["velocity_self_transition"],)

    tool_onehot_msa(
        adata,
        dirname=f'target/evolocity_alignments/{namespace}',
        n_threads=40,
    )
    tool_residue_scores(adata)
    plot_residue_scores(adata, save=f'_{namespace}_residue_scores.png')

    import scvelo as scv
    scv.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    scv.pl.velocity_embedding(
        adata, basis='umap', color='tax_group',
        save=f'_{namespace}_taxonomy_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=1., smooth=1.,
        arrow_size=1., arrow_length=3.,
        color='tax_group', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/scvelo__{namespace}_taxonomy_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=1., smooth=1., linewidth=1.,
        color='tax_group', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/scvelo__{namespace}_taxonomy_velostream.png', dpi=500)
    plt.close()

    ax = plot_path(
        adata,
        source_idx=list(adata.obs['gene_id']).index('CYC_HUMAN'),
        target_idx=list(adata.obs['gene_id']).index('CYC1_YEAST'),
    )
    ax = plot_path(
        adata,
        source_idx=list(adata.obs['gene_id']).index('CYC_APIME'),
        target_idx=list(adata.obs['gene_id']).index('CYC1_YEAST'),
        ax=ax,
    )
    ax = plot_path(
        adata,
        source_idx=list(adata.obs['gene_id']).index('CYC_MAIZE'),
        target_idx=list(adata.obs['gene_id']).index('CYC1_YEAST'),
        ax=ax,
    )

    plot_pseudotime(
        adata, basis='umap', min_mass=1., smooth=0.6, levels=100,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False, ax=ax,
        rank_transform=True,
        save=f'_{namespace}_pseudotime.png', dpi=500
    )

    scv.pl.scatter(adata, color=[ 'root_cells', 'end_points' ],
                   cmap=plt.cm.get_cmap('magma').reversed(),
                   save=f'_{namespace}_origins.png', dpi=500)

    plt.figure()
    sns.violinplot(data=adata.obs, x='tax_group', y='pseudotime',
                   order=[
                       'eukaryota',
                       'fungi',
                       'viridiplantae',
                       'arthropoda',
                       'chordata',
                       'mammalia',
                       'primate',
                   ])
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(f'figures/{namespace}_taxonomy_pseudotime.png', dpi=500)
    plt.close()

    sc.pl.umap(adata, color='pseudotime', edges=True, cmap='magma',
               save=f'_{namespace}_pseudotime.png')

    nnan_idx = (np.isfinite(adata.obs['homology']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-homology Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['homology'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-homology Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['homology'][nnan_idx])))

    anc_seq = """
    MKNFFFCLFIPYIMPPKARAPLPPGDAERGKKLFKSRA
    AQCHTAEKGGANSTGPNLYGLFGRKSGTVPGYAYSNAN
    KNAGIVWEEETLHEYLENPKKYVPGTKMAFAGIKKPKD
    RADIIAYLKTLKDEELS
    """.replace('\n', '').replace(' ', '')
    adata.obs['anc_dist'] = [
        fuzz.ratio(anc_seq, seq) for seq in adata.obs['seq']
    ]

    nnan_idx = (np.isfinite(adata.obs['anc_dist']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-ancestral dist Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['anc_dist'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-ancestral dist Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['anc_dist'][nnan_idx])))

    adata.write(f'target/results/{namespace}_adata.h5ad')

if __name__ == '__main__':
    args = parse_args()

    namespace = args.namespace
    if args.model_name == 'tape':
        namespace += '_tape'

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z'
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

    model, seqs = setup(args)

    if 'esm' in args.model_name:
        vocabulary = { tok: model.alphabet_.tok_to_idx[tok]
                       for tok in model.alphabet_.tok_to_idx
                       if '<' not in tok and tok != '.' and tok != '-' }
        args.checkpoint = args.model_name
    elif args.model_name == 'tape':
        vocabulary = { tok: model.alphabet_[tok]
                       for tok in model.alphabet_ if '<' not in tok }
        args.checkpoint = args.model_name
    elif args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.train or args.train_split or args.test:
        train_test(args, model, seqs, vocabulary, split_seqs)

    if args.ancestral:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        tprint('Ancestral analysis...')
        cyc_ancestral(args, model, seqs, vocabulary, namespace=namespace)

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        tprint('All cytochrome c sequencecs:')
        evo_cyc(args, model, seqs, vocabulary, namespace=namespace)

        if args.model_name != 'tape':
            tprint('Restrict based on similarity to training:')
            evo_cyc(args, model, seqs, vocabulary, namespace='cyc_homologous')
