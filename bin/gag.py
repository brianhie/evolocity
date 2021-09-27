from mutation import *
from evolocity_graph import *
import evolocity as evo

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='HIV gag sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., esm1b, tape)')
    parser.add_argument('--namespace', type=str, default='gag',
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

def load_meta(meta_fnames):
    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                accession = line[1:].rstrip()
                fields = line.rstrip().split('.')
                subtype, country, year, strain = (
                    fields[0], fields[1], fields[2], fields[3]
                )

                if year == '-':
                    year = None
                else:
                    year = int(year)

                subtype = subtype.split('_')[-1]
                subtype = subtype.lstrip('>0123')

                keep_subtypes = {
                    'A', 'A1', 'A1A2', 'A1C', 'A1D', 'A2', 'A3', 'A6',
                    'AE', 'AG', 'B', 'C', 'BC', 'D',
                    'F', 'F1', 'F2', 'G', 'H', 'J',
                    'K', 'L', 'N', 'O', 'P', 'U',
                }
                if subtype not in keep_subtypes:
                    subtype = 'Other'

                metas[accession] = {
                    'gene_id': accession,
                    'subtype': subtype,
                    'country': country,
                    'year': year,
                    'strain': strain,
                    'accession': fields[-1],
                }
    return metas

def process(args, fnames, meta_fnames):
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            accession = record.description
            meta = metas[accession]
            meta['seqlen'] = len(str(record.seq))
            if meta['seqlen'] < 480 or meta['seqlen'] > 515:
                continue
            if 'X' in record.seq:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            seqs[record.seq].append(meta)

    tprint('Found {} unique sequences'.format(len(seqs)))

    seqs = training_distances(seqs, namespace=args.namespace)

    return seqs

def setup(args):
    fnames = [ 'data/gag/gag.fa' ]
    meta_fnames = [ 'data/gag/gag.fa' ]

    import pickle
    cache_fname = 'target/ev_cache/gag_seqs.pkl'
    try:
        with open(cache_fname, 'rb') as f:
            seqs = pickle.load(f)
    except:
        seqs = process(args, fnames, meta_fnames)
        with open(cache_fname, 'wb') as of:
            pickle.dump(seqs, of)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1000)

    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'year', 'country', 'subtype' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

    cluster2subtype = {}
    for i in range(len(adata)):
        cluster = adata.obs['louvain'][i]
        if cluster not in cluster2subtype:
            cluster2subtype[cluster] = []
        cluster2subtype[cluster].append(adata.obs['subtype'][i])
    largest_pct_subtype = []
    for cluster in cluster2subtype:
        count = Counter(cluster2subtype[cluster]).most_common(1)[0][1]
        pct_subtype = float(count) / len(cluster2subtype[cluster])
        largest_pct_subtype.append(pct_subtype)
        tprint('\tCluster {}, largest subtype % = {}'
               .format(cluster, pct_subtype))
    tprint('Purity, Louvain and subtype: {}'
           .format(np.mean(largest_pct_subtype)))

def plot_umap(adata):
    sc.pl.umap(adata, color='subtype', save='_gag_subtype.png')
    sc.pl.umap(adata, color='year', save='_gag_year.png')
    sc.pl.umap(adata, color='homology', save='_gag_homology.png')
    sc.pl.umap(adata, color='louvain', save='_gag_louvain.png')

def seqs_to_anndata(seqs):
    keys = set([ key for seq in seqs for meta in seqs[seq] for key in meta ])

    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    for seq in seqs:
        X.append(seqs[seq][0]['embedding'])
        for key in keys:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            values = [ meta[key] for meta in seqs[seq] if key in meta ]
            if len(values) > 0:
                obs[key].append(Counter(values).most_common(1)[0][0])
            else:
                obs[key].append(None)
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    return adata

def gag_siv_cpz(args, model, seqs, vocabulary, namespace='glo'):
    path_fname = 'data/gag/gag_cpz.fa'
    nodes = [
        (record.id, str(record.seq))
        for record in SeqIO.parse(path_fname, 'fasta')
    ]

    keep_subtypes = {
        'AE', 'B', 'C', 'BC', 'D',
    }

    dist_data = []
    for idx, (full_name, seq) in enumerate(nodes):
        for uniprot_seq in seqs:
            name = full_name.split('(')[-1].split(')')[0]
            gag_type = Counter([
                meta['subtype'] for meta in seqs[uniprot_seq]
            ]).most_common(1)[0][0]
            if gag_type not in keep_subtypes:
                if 'A' in gag_type:
                    gag_type = 'A'
                else:
                    continue
            score = likelihood_muts(seq, uniprot_seq,
                                    args, vocabulary, model,)
            homology = fuzz.ratio(seq, uniprot_seq)
            dist_data.append([ gag_type, name, score, homology, 'human' ])

    df = pd.DataFrame(dist_data, columns=[
        'gag_type', 'name', 'score', 'homology', 'host'
    ])

    plot_ancestral(df, meta_key='gag_type', namespace=namespace)
    plot_ancestral(df, meta_key='name', name_key='gag_type', namespace=namespace)
    plot_ancestral(df, meta_key='name', name_key='host', namespace=namespace)

def evo_gag(args, model, seqs, vocabulary, namespace='gag'):
    if args.model_name != 'esm1b':
        namespace += f'_{args.model_name}'
    if args.velocity_score != 'lm':
        namespace += f'_{args.velocity_score}'

    #############################
    ## Visualize Gag landscape ##
    #############################

    adata_cache = 'target/ev_cache/gag_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        seqs = populate_embedding(args, model, seqs, vocabulary, use_cache=True)
        adata = seqs_to_anndata(seqs)
        sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)
        adata.write(adata_cache)

    if 'homologous' in namespace:
        adata = adata[adata.obs['homology'] > 80.]
        sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

    if '_onehot' in namespace:
        evo.tl.onehot_msa(
            adata,
            dirname=f'target/evolocity_alignments/{namespace}',
            n_threads=40,
        )
        sc.pp.neighbors(adata, n_neighbors=40, metric='manhattan',
                        use_rep='X_onehot')
        sc.tl.umap(adata)

    keep_subtypes = {
        'AE', 'B', 'C', 'BC', 'D', 'E',
    }
    adata.obs['simple_subtype'] = [
        subtype if subtype in keep_subtypes else (
            'A' if 'A' in subtype else 'Other'
        ) for subtype in adata.obs['subtype']
    ]

    tprint('Analyzing {} sequences...'.format(adata.X.shape[0]))
    evo.set_figure_params(dpi_save=500)
    plot_umap(adata)

    cache_prefix = f'target/ev_cache/{namespace}_knn40'
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
        adata, basis='umap', color='year',
        save=f'_{namespace}_year_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=1., smooth=1.,
        arrow_size=1., arrow_length=3.,
        color='year', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_year_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=3., smooth=1., linewidth=0.7,
        color='year', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#dddddd')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_year_velostream.png', dpi=500)
    plt.close()
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=3., smooth=1., linewidth=0.7,
        color='simple_subtype', show=False, legend_loc='lower right',
        palette=[ '#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#9467bd', '#8c564b',
                  '#888888'],
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#dddddd')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_subtype_velostream.png', dpi=500)
    plt.close()

    plt.figure()
    ax = evo.pl.velocity_contour(
        adata,
        basis='umap', smooth=0.8, pf_smooth=1., levels=100,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.savefig(f'figures/evolocity__{namespace}_contour.png', dpi=500)
    plt.close()

    sc.pl.umap(adata, color=[ 'root_nodes', 'end_points' ],
               edges=True, edges_color='#aaaaaa',
               color_map=plt.cm.get_cmap('magma').reversed(),
               save=f'_{namespace}_origins.png')
    sc.pl.umap(adata, color='pseudotime', edges=True, edges_color='#aaaaaa',
               color_map='inferno', save=f'_{namespace}_pseudotime.png')

    plt.figure(figsize=(3, 6))
    sns.boxplot(data=adata.obs, x='simple_subtype', y='pseudotime',
                order=[
                    'A',
                    'AE',
                    'B',
                    'C',
                    'BC',
                ])
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(f'figures/{namespace}_subtype_pseudotime.svg', dpi=500)
    plt.close()

    nnan_idx = (np.isfinite(adata.obs['year']) &
                np.isfinite(adata.obs['pseudotime']))
    tprint('Pseudotime-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'][nnan_idx],
                                 adata.obs['year'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudotime-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudotime'][nnan_idx],
                                adata.obs['year'][nnan_idx])))

    with open(f'target/ev_cache/{namespace}_pseudotime.txt', 'w') as of:
        of.write('\n'.join([ str(x) for x in adata.obs['pseudotime'] ]) + '\n')

    if args.model_name != 'tape':
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
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
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

        tprint('SIV analysis...')
        gag_siv_cpz(args, model, seqs, vocabulary, namespace=namespace)

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))

        evo_gag(args, model, seqs, vocabulary, namespace=namespace)

        if args.model_name == 'esm1b' and args.velocity_score == 'lm':
            tprint('Restrict based on similarity to training:')
            evo_gag(args, model, seqs, vocabulary, namespace='gag_homologous')

            tprint('One hot featurization:')
            evo_gag(args, model, seqs, vocabulary, namespace='gag_onehot')
