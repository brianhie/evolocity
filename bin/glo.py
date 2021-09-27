from mutation import *
from evolocity_graph import *
import evolocity as evo

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Globin sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='glo',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--ancestral', action='store_true',
                        help='Analyze ancestral sequences')
    parser.add_argument('--evolocity', action='store_true',
                        help='Analyze evolocity')
    parser.add_argument('--velocity-score', type=str, default='lm',
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
        (_, accession, gene_id, species, species_id,
         gene_symbol, pe, sv) = record.split('|')
    else:
        (_, accession, gene_id, species, species_id,
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
    assert(tax_group is not None)

    gene_id = gene_id.lower()
    globin_type = 'other/unlabeled'
    if 'myoglobin' in gene_id:
        globin_type = 'myoglobin'
    elif 'neuroglobin' in gene_id:
        globin_type = 'neuroglobin'
    elif 'cytoglobin' in gene_id:
        globin_type = 'cytoglobin'
    elif 'alpha' in gene_id or \
         'hba' in gene_id:
        globin_type = 'hemoglobin_alpha'
    elif 'hbb' in gene_id or \
         'beta' in gene_id:
        globin_type = 'hemoglobin_beta'

    return {
        'accession': accession,
        'gene_id': gene_id,
        'species': species[3:],
        'tax_id': tax_id,
        'tax_group': tax_group,
        'lineage': lineage,
        'globin_type': globin_type,
        'gene_symbol': gene_symbol[3:] if gene_symbol is not None else None,
        'pe': pe[3:],
        'sv': sv[3:],
    }

def process(fnames):
    taxonomy = load_taxonomy()

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 135 or len(record.seq) > 155:
                continue
            meta = parse_meta(record.id, taxonomy)
            if 'Eukaryota' not in meta['lineage']:
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
    fnames = [ 'data/glo/uniprot_globin.fa' ]

    import pickle
    cache_fname = 'target/ev_cache/glo_seqs.pkl'
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

def plot_umap(adata, namespace='glo'):
    sc.pl.umap(adata, color='tax_group', edges=True,
               save='_{}_taxonomy.png'.format(namespace))
    sc.pl.umap(adata, color='globin_type', edges=True,
               save='_{}_glotype.png'.format(namespace))
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

def globin_ancestral(args, model, seqs, vocabulary, namespace='glo'):
    path_fname = 'data/glo/ancestral_globins.fa'
    nodes = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(path_fname, 'fasta')
    ]

    ######################################
    ## See how local likelihoods change ##
    ######################################

    glo_types = {
        'myoglobin',
        'hemoglobin_alpha',
        'hemoglobin_beta'
    }

    dist_data = []
    for idx, (name, seq) in enumerate(nodes):
        for uniprot_seq in seqs:
            glo_type = Counter([
                meta['globin_type'] for meta in seqs[uniprot_seq]
            ]).most_common(1)[0][0]
            if glo_type not in glo_types:
                continue
            if 'myo' not in name and glo_type == 'myoglobin':
                continue
            if name == 'ancestral_beta_hemoglobin' and 'alpha' in glo_type:
                continue
            if name == 'ancestral_alpha_hemoglobin' and 'beta' in glo_type:
                continue
            score = likelihood_muts(seq, uniprot_seq,
                                    args, vocabulary, model,)
            homology = fuzz.ratio(seq, uniprot_seq)
            dist_data.append([ glo_type, name, score, homology ])

    df = pd.DataFrame(dist_data, columns=[
        'glo_type', 'name', 'score', 'homology'
    ])

    plot_ancestral(df, meta_key='glo_type', namespace=namespace)
    plot_ancestral(df, meta_key='name', name_key='glo_type', namespace=namespace)

def globin_paths(path_fname, args, model, seqs, vocabulary, namespace='glo'):
    tprint(f'Path defined in {path_fname}:')

    nodes = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(path_fname, 'fasta')
    ]

    ######################################
    ## See how local likelihoods change ##
    ######################################

    data, dist_data = [], []
    for idx, (name, seq) in enumerate(nodes):
        if idx == 0:
            continue
        seq_prev = nodes[idx - 1][1]
        score_full = likelihood_full(seq_prev, seq,
                                     args, vocabulary, model,)
        score_muts = likelihood_muts(seq_prev, seq,
                                     args, vocabulary, model,)
        score_self = likelihood_self(seq_prev, seq,
                                     args, vocabulary, model,)
        data.append([ name, seq,
                      score_full, score_muts, score_self ])
        tprint('{}: {}, {}, {}'.format(
            name, score_full, score_muts, score_self
        ))

    df = pd.DataFrame(data, columns=[ 'name', 'seq', 'full', 'muts',
                                      'self_score' ])
    tprint('Sum of full scores: {}'.format(sum(df.full)))
    tprint('Sum of local scores: {}'.format(sum(df.muts)))
    tprint('Sum of self scores: {}'.format(sum(df.self_score)))

def evo_globin(args, model, seqs, vocabulary, namespace='glo'):
    if args.model_name != 'esm1b':
        namespace += f'_{args.model_name}'
    if args.velocity_score != 'lm':
        namespace += f'_{args.velocity_score}'

    #########################
    ## Visualize landscape ##
    #########################

    adata_cache = 'target/ev_cache/glo_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        seqs = populate_embedding(args, model, seqs, vocabulary, use_cache=True)
        adata = seqs_to_anndata(seqs)
        sc.pp.neighbors(adata, n_neighbors=50, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=0.5)
        adata.write(adata_cache)

    if 'homologous' in namespace:
        adata = adata[adata.obs['homology'] > 80.]
        sc.pp.neighbors(adata, n_neighbors=50, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=0.5)

    if '_onehot' in namespace:
        evo.tl.onehot_msa(
            adata,
            dirname=f'target/evolocity_alignments/{namespace}',
            n_threads=40,
        )
        sc.pp.neighbors(adata, n_neighbors=50, metric='manhattan',
                        use_rep='X_onehot')
        sc.tl.umap(adata)

    tprint('Analyzing {} sequences...'.format(adata.X.shape[0]))
    evo.set_figure_params(dpi_save=500)
    plot_umap(adata, namespace=namespace)

    #####################################
    ## Compute evolocity and visualize ##
    #####################################

    cache_prefix = f'target/ev_cache/{namespace}_knn50'
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
        evo.tl.velocity_graph(adata, model_name=args.model_name,
                              score=args.velocity_score)
        from scipy.sparse import save_npz
        save_npz('{}_vgraph.npz'.format(cache_prefix),
                 adata.uns["velocity_graph"],)
        save_npz('{}_vgraph_neg.npz'.format(cache_prefix),
                 adata.uns["velocity_graph_neg"],)
        np.save('{}_vself_transition.npy'.format(cache_prefix),
                adata.obs["velocity_self_transition"],)

    evo.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    evo.pl.velocity_embedding(
        adata, basis='umap', color='tax_group',
        save=f'_{namespace}_taxonomy_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=1., smooth=1.,
        arrow_size=1., arrow_length=3.,
        color='globin_type', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#dddddd')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_glotype_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=4., smooth=1.5, linewidth=1.,
        color='globin_type', legend_loc=False,
        palette=[ '#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#888888', ],
        show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#dddddd')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_glotype_velostream.png', dpi=500)
    plt.close()

    evo.pl.velocity_contour(
        adata, basis='umap', smooth=0.6, levels=100,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
        save=f'_{namespace}_contour.png', dpi=500
    )

    sc.pl.umap(adata, color=[ 'root_nodes', 'end_points' ],
               cmap=plt.cm.get_cmap('magma').reversed(),
               save=f'_{namespace}_origins.png')

    tax_order = [
        'eukaryota',
        'viridiplantae',
        'fungi',
        'arthropoda',
        'chordata',
        'mammalia',
        'primate',
    ]
    plt.figure()
    sns.boxplot(
        data=adata.obs, x='tax_group', y='pseudotime',
        order=tax_order,
    )
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(f'figures/{namespace}_taxonomy_pseudotime.svg')
    plt.close()

    type_order = [
        'neuroglobin',
        'cytoglobin',
        'myoglobin',
        'hemoglobin_alpha',
        'hemoglobin_beta',
    ]
    plt.figure()
    sns.boxplot(
        data=adata.obs, x='globin_type', y='pseudotime',
        order=type_order,
    )
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(f'figures/{namespace}_type_pseudotime.svg')
    plt.close()

    sc.pl.umap(adata, color='pseudotime', edges=True, cmap='inferno',
               save=f'_{namespace}_pseudotime.png')

    with open(f'target/ev_cache/{namespace}_pseudotime.txt', 'w') as of:
        of.write('\n'.join([ str(x) for x in adata.obs['pseudotime'] ]) + '\n')

    nnan_idx = (np.isfinite(adata.obs['homology']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-homology Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['homology'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-homology Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['homology'][nnan_idx])))

    seqlens = [ len(seq) for seq in adata.obs['seq'] ]
    tprint('Pseudotime-length Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'], seqlens,
                                 nan_policy='omit')))


if __name__ == '__main__':
    args = parse_args()

    namespace = args.namespace

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

    if args.ancestral:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        tprint('Ancestral analysis...')
        globin_ancestral(args, model, seqs, vocabulary, namespace=namespace)

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        tprint('All globin sequences:')
        evo_globin(args, model, seqs, vocabulary, namespace=namespace)

        if args.model_name == 'esm1b' and args.velocity_score == 'lm':
            tprint('Restrict based on similarity to training:')
            evo_globin(args, model, seqs, vocabulary, namespace='glo_homologous')

            tprint('One hot featurization:')
            evo_globin(args, model, seqs, vocabulary, namespace='glo_onehot')
