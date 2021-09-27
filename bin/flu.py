from mutation import *
from evolocity_graph import *
import evolocity as evo

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Flu sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='flu',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=14,
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
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--semantics', action='store_true',
                        help='Analyze mutational semantic change')
    parser.add_argument('--combfit', action='store_true',
                        help='Analyze combinatorial fitness')
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
            header = f.readline().rstrip().split('\t')
            for line in f:
                fields = line.rstrip().split('\t')
                accession = fields[1]
                meta = {}
                for key, value in zip(header, fields):
                    if key == 'Subtype':
                        meta[key] = value.strip('()').split('N')[0].split('/')[-1]
                    elif key == 'Collection Date':
                        meta[key] = int(value.split('/')[-1]) \
                                    if value != '-N/A-' else None
                    elif key == 'Host Species':
                        meta[key] = value.split(':')[1].split('/')[-1].lower()
                    else:
                        meta[key] = value
                meta['gene_id'] = '{}_{}_{}'.format(
                    meta['Subtype'], meta['Collection Date'], meta['Host Species']
                )
                metas[accession] = meta
    return metas

def process(args, fnames, meta_fnames):
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if 'Reference_Perth2009_HA_coding_sequence' in record.description:
                continue
            if str(record.seq).count('X') > 10:
                continue
            accession = record.description.split('|')[0].split(':')[1]
            if metas[accession]['Host Species'] != 'human':
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            seqs[record.seq].append(metas[accession])

    seqs = training_distances(seqs, namespace=args.namespace)

    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs, val_seqs = {}, {}, {}

    old_cutoff = 1990
    new_cutoff = 2018

    tprint('Splitting seqs...')
    for seq in seqs:
        # Pick validation set based on date.
        seq_dates = [
            meta['Collection Date'] for meta in seqs[seq]
            if meta['Collection Date'] is not None
        ]
        if len(seq_dates) > 0:
            oldest_date = sorted(seq_dates)[0]
            if oldest_date < old_cutoff or oldest_date >= new_cutoff:
                test_seqs[seq] = seqs[seq]
                continue
        train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies.fa' ]
    meta_fnames = [ 'data/influenza/ird_influenzaA_HA_allspecies_meta.tsv' ]

    import pickle
    cache_fname = 'target/ev_cache/flu_seqs.pkl'
    try:
        with open(cache_fname, 'rb') as f:
            seqs = pickle.load(f)
    except:
        seqs = process(args, fnames, meta_fnames)
        with open(cache_fname, 'wb') as of:
            pickle.dump(seqs, of)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'Collection Date', 'Country', 'Subtype',
                     'Flu Season', 'Host Species', 'Strain Name' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

    cluster2subtype = {}
    cluster2species = {}
    for i in range(len(adata)):
        cluster = adata.obs['louvain'][i]
        if cluster not in cluster2subtype:
            cluster2subtype[cluster] = []
            cluster2species[cluster] = []
        cluster2subtype[cluster].append(adata.obs['Subtype'][i])
        cluster2species[cluster].append(adata.obs['Host Species'][i])
    largest_pct_subtype = []
    largest_pct_species = []
    for cluster in cluster2subtype:
        count = Counter(cluster2subtype[cluster]).most_common(1)[0][1]
        largest_pct_subtype.append(float(count) /
                                   len(cluster2subtype[cluster]))
        count = Counter(cluster2species[cluster]).most_common(1)[0][1]
        largest_pct_species.append(float(count) /
                                   len(cluster2species[cluster]))


    for idx, pct in enumerate(largest_pct_subtype):
        tprint('\tCluster {}, largest subtype % = {}'.format(idx, pct))
    for idx, pct in enumerate(largest_pct_species):
        tprint('\tCluster {}, largest species % = {}'.format(idx, pct))

    tprint('Purity, Louvain and subtype: {}'
           .format(np.mean(largest_pct_subtype)))
    tprint('Purity, Louvain and host species: {}'
           .format(np.mean(largest_pct_species)))

def seq_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        counts = Counter(adata_cluster.obs['seq'])
        with open('target/flu/clusters/cluster{}.fa'.format(cluster), 'w') as of:
            for i, (seq, count) in enumerate(counts.most_common()):
                of.write('>cluster{}_{}_{}\n'.format(cluster, i, count))
                of.write(seq + '\n\n')

def plot_umap(adata, namespace='flu'):
    sc.pl.umap(adata, color='Subtype',
               save='_{}_subtype.png'.format(namespace))
    sc.pl.umap(adata, color='Collection Date',
               edges=True, edges_color='#aaaaaa',
               save='_{}_date.png'.format(namespace))
    sc.pl.umap(adata, color='louvain',
               save='_{}_louvain.png'.format(namespace))
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

def analyze_embedding(args, model, seqs, vocabulary):
    seqs = populate_embedding(args, model, seqs, vocabulary,
                              use_cache=True)

    adata = seqs_to_anndata(seqs)

    adata = adata[
        np.logical_or.reduce((
            adata.obs['Host Species'] == 'human',
            adata.obs['Host Species'] == 'avian',
            adata.obs['Host Species'] == 'swine',
        ))
    ]

    sc.pp.neighbors(adata, n_neighbors=100, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)

    sc.tl.umap(adata, min_dist=1.)
    plot_umap(adata)
    plot_umap(adata[adata.obs['louvain'] == '30'],
              namespace='flu1918')

    interpret_clusters(adata)

    seq_clusters(adata)

def evo_ha(args, model, seqs, vocabulary, namespace='h1'):
    if args.model_name != 'esm1b':
        namespace += f'_{args.model_name}'
    if args.velocity_score != 'lm':
        namespace += f'_{args.velocity_score}'

    ############################
    ## Visualize HA landscape ##
    ############################

    adata_cache = 'target/ev_cache/h1_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        seqs = populate_embedding(args, model, seqs, vocabulary,
                                  use_cache=True)

        adata = seqs_to_anndata(seqs)

        adata = adata[(adata.obs['Host Species'] == 'human') &
                      (adata.obs['Subtype'] == 'H1')]

        sc.pp.neighbors(adata, n_neighbors=50, use_rep='X')
        sc.tl.umap(adata, min_dist=1.)
        sc.tl.louvain(adata, resolution=1.)
        adata.write(adata_cache)

    if 'homologous' in namespace:
        adata = adata[adata.obs['homology'] > 80.]
        sc.pp.neighbors(adata, n_neighbors=50, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

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
        adata, basis='umap', color='Collection Date',
        save=f'_{namespace}_year_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=4., smooth=1.,
        arrow_size=2., arrow_length=3.,
        color='Collection Date', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_year_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=4., smooth=1., linewidth=0.7,
        color='Collection Date', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_year_velostream.png', dpi=500)
    plt.close()

    # Evolocity pseudotime visualization.

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
               cmap=plt.cm.get_cmap('magma').reversed(),
               save=f'_{namespace}_origins.png')

    sc.pl.umap(adata, color='pseudotime', edges=True,
               edges_color='#aaaaaa', cmap='inferno',
               save=f'_{namespace}_pseudotime.png')

    nnan_idx = (np.isfinite(adata.obs['Collection Date']) &
                np.isfinite(adata.obs['pseudotime']))

    adata_nnan = adata[nnan_idx]

    plt.figure()
    sns.regplot(x='Collection Date', y='pseudotime',
                     data=adata_nnan.obs, ci=None)
    plt.savefig(f'figures/{namespace}_pseudotime-time.png', dpi=500)
    plt.close()

    with open(f'target/ev_cache/{namespace}_pseudotime.txt', 'w') as of:
        of.write('\n'.join([ str(x) for x in adata.obs['pseudotime'] ]) + '\n')

    tprint('Pseudotime-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata_nnan.obs['pseudotime'],
                                 adata_nnan.obs['Collection Date'],
                                 nan_policy='omit')))
    tprint('Pseudotime-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata_nnan.obs['pseudotime'],
                                adata_nnan.obs['Collection Date'])))

    seqlens = [ len(seq) for seq in adata_nnan.obs['seq'] ]
    tprint('Pseudotime-length Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata_nnan.obs['pseudotime'], seqlens,
                                 nan_policy='omit')))
    tprint('Pseudotime-length Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata_nnan.obs['pseudotime'], seqlens)))

    if args.model_name != 'tape':
        nnan_idx = (np.isfinite(adata_nnan.obs['homology']) &
                    np.isfinite(adata_nnan.obs['pseudotime']))
        tprint('Pseudotime-homology Spearman r = {}, P = {}'
               .format(*ss.spearmanr(adata_nnan.obs['pseudotime'],
                                     adata_nnan.obs['homology'],
                                     nan_policy='omit')))
        tprint('Pseudotime-homology Pearson r = {}, P = {}'
               .format(*ss.pearsonr(adata.obs['pseudotime'],
                                    adata.obs['homology'])))


if __name__ == '__main__':
    args = parse_args()

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

    if args.embed:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))
        analyze_embedding(args, model, seqs, vocabulary)

    if args.semantics:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        from escape import load_doud2018, load_lee2019

        tprint('Lee et al. 2018...')
        seq_to_mutate, escape_seqs = load_doud2018()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          beta=1., plot_acquisition=True,
                          plot_namespace='flu_h1')
        tprint('')

        tprint('Lee et al. 2019...')
        seq_to_mutate, escape_seqs = load_lee2019()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          beta=1., plot_acquisition=True,
                          plot_namespace='flu_h3')

    if args.combfit:
        from combinatorial_fitness import load_doud2016
        tprint('Doud et al. 2016...')
        wt_seqs, seqs_fitness = load_doud2016()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 beta=1.)

        from combinatorial_fitness import load_wu2020
        tprint('Wu et al. 2020...')
        wt_seqs, seqs_fitness = load_wu2020()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 beta=1.)

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        namespace = 'h1'
        evo_ha(args, model, seqs, vocabulary, namespace=namespace)

        if args.model_name == 'esm1b' and args.velocity_score == 'lm':
            tprint('Restrict based on similarity to training:')
            evo_ha(args, model, seqs, vocabulary, namespace='h1_homologous')

            tprint('One hot featurization:')
            evo_ha(args, model, seqs, vocabulary, namespace='h1_onehot')
