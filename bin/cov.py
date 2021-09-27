from mutation import *
from evolocity_graph import *
import evolocity as evo

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Coronavirus sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='cov',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=11,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--evolocity', action='store_true',
                        help='Analyze evolocity')
    parser.add_argument('--velocity-score', type=str, default='lm',
                        help='Analyze evolocity')
    args = parser.parse_args()
    return args

def parse_gisaid(entry):
    fields = entry.split('|')

    type_id = fields[1].split('/')[1]

    if type_id in { 'bat', 'canine', 'cat', 'env', 'mink',
                    'pangolin', 'tiger' }:
        host = type_id
        country = 'NA'
        continent = 'NA'
    else:
        host = 'human'
        from locations import country2continent
        country = type_id
        if type_id in country2continent:
            continent = country2continent[country]
        else:
            continent = 'NA'

    from mammals import species2group

    date = fields[2]
    date = date.replace('00', '01')
    timestamp = time.mktime(dparse(date).timetuple())

    meta = {
        'gene_id': fields[1],
        'date': date,
        'timestamp': timestamp,
        'host': host,
        'group': species2group[host].lower(),
        'country': country,
        'continent': continent,
        'dataset': 'gisaid',
    }
    return meta

def process(fnames):
    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if len(record.seq) < 1263 or len(record.seq) > 1283:
                continue
            if str(record.seq).count('X') > 0:
                continue
            if not record.seq.startswith('M') or not record.seq.endswith('HYT*'):
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            meta = parse_gisaid(record.description)
            meta['accession'] = record.description
            seqs[record.seq].append(meta)
    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs = {}, {}

    tprint('Splitting seqs...')
    for idx, seq in enumerate(seqs):
        if idx % 10 < 2:
            test_seqs[seq] = seqs[seq]
        else:
            train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [
        #'data/cov/spikeprot0527.fasta',
        'data/cov/spikeprot0825.fa',
    ]

    import pickle
    cache_fname = 'target/ev_cache/cov_seqs.pkl'
    try:
        with open(cache_fname, 'rb') as f:
            seqs = pickle.load(f)
    except:
        seqs = process(fnames)
        with open(cache_fname, 'wb') as of:
            pickle.dump(seqs, of)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1200)

    return model, seqs

def interpret_clusters(adata):
    clusters = sorted(set(adata.obs['louvain']))
    for cluster in clusters:
        tprint('Cluster {}'.format(cluster))
        adata_cluster = adata[adata.obs['louvain'] == cluster]
        for var in [ 'host', 'continent', 'gene_id' ]:
            tprint('\t{}:'.format(var))
            counts = Counter(adata_cluster.obs[var])
            for val, count in counts.most_common():
                tprint('\t\t{}: {}'.format(val, count))
        tprint('')

def plot_umap(adata, namespace='cov'):
    categories = [
        'timestamp',
        'louvain',
        'seqlen',
        'continent',
        'n_seq',
        'host',
    ]
    for category in categories:
        sc.pl.umap(adata, color=category,
                   edges=True, edges_color='#aaaaaa',
                   save='_{}_{}.png'.format(namespace, category))

def seqs_to_anndata(seqs):
    X, obs = [], {}
    obs['n_seq'] = []
    obs['seq'] = []
    obs['seqlen'] = []
    for seq in seqs:
        meta = seqs[seq][0]
        X.append(meta['embedding'])
        earliest_idx = np.argmin([
            meta['timestamp'] for meta in seqs[seq]
        ])
        for key in meta:
            if key == 'embedding':
                continue
            if key not in obs:
                obs[key] = []
            obs[key].append([
                meta[key] for meta in seqs[seq]
            ][earliest_idx])
        obs['n_seq'].append(len(seqs[seq]))
        obs['seq'].append(str(seq))
        obs['seqlen'].append(len(seq))
    X = np.array(X)

    adata = AnnData(X)
    for key in obs:
        adata.obs[key] = obs[key]

    return adata

def spike_evolocity(args, model, seqs, vocabulary, namespace='cov'):
    if args.model_name != 'esm1b':
        namespace += f'_{args.model_name}'
    if args.velocity_score != 'lm':
        namespace += f'_{args.velocity_score}'

    ###############################
    ## Visualize Spike landscape ##
    ###############################

    adata_cache = 'target/ev_cache/cov_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
        seqs = populate_embedding(args, model, seqs, vocabulary,
                                  use_cache=True)
        adata = seqs_to_anndata(seqs)
        adata = adata[
            adata.obs['timestamp'] >
            time.mktime(dparse('2019-11-30').timetuple())
        ]

        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=0.3)

        adata.write(adata_cache)

    adata.obs['seq'] = [ seq.rstrip('*') for seq in adata.obs['seq'] ]

    if '_onehot' in namespace:
        evo.tl.onehot_msa(
            adata,
            dirname=f'target/evolocity_alignments/{namespace}',
            n_threads=40,
        )
        sc.pp.neighbors(adata, n_neighbors=30, metric='manhattan',
                        use_rep='X_onehot')
        sc.tl.umap(adata)

    tprint('Analyzing {} sequences...'.format(adata.X.shape[0]))
    evo.set_figure_params(dpi_save=500, figsize=(5, 4))
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
        evo.tl.velocity_graph(adata, model_name=args.model_name,
                              score=args.velocity_score)
        from scipy.sparse import save_npz
        save_npz('{}_vgraph.npz'.format(cache_prefix),
                 adata.uns["velocity_graph"],)
        save_npz('{}_vgraph_neg.npz'.format(cache_prefix),
                 adata.uns["velocity_graph_neg"],)
        np.save('{}_vself_transition.npy'.format(cache_prefix),
                adata.obs["velocity_self_transition"],)

    wt_fname = 'data/cov/cov2_spike_wt.fasta'
    wt_seq = str(SeqIO.read(wt_fname, 'fasta').seq)

    if args.model_name == 'esm1b' and args.velocity_score == 'lm':
        evo.tl.onehot_msa(
            adata,
            reference=list(adata.obs['seq']).index(wt_seq),
            dirname=f'target/evolocity_alignments/{namespace}',
            n_threads=40,
        )
        evo.tl.residue_scores(adata)
        evo.pl.residue_scores(
            adata,
            percentile_keep=0,
            save=f'_{namespace}_residue_scores.png',
        )
        evo.pl.residue_categories(
            adata,
            positions=[ 17, 153, 416, 451, 477, 483, 500, 613, 680, 949, ],
            namespace=namespace,
            reference=list(adata.obs['seq']).index(wt_seq),
        )

    evo.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    evo.pl.velocity_embedding(
        adata, basis='umap', color='timestamp',
        save=f'_{namespace}_time_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=1., smooth=1.2,
        arrow_size=1., arrow_length=3.,
        color='timestamp', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_time_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=1., smooth=1., density=1.2,
        color='timestamp', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'figures/evolocity__{namespace}_time_velostream.png', dpi=500)
    plt.close()

    plt.figure()
    ax = evo.pl.velocity_contour(
        adata,
        basis='umap', smooth=1., pf_smooth=1.5, levels=100,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.savefig(f'figures/evolocity__{namespace}_pseudotime.png', dpi=500)
    plt.close()

    sc.pl.umap(adata, color=[ 'root_nodes', 'end_points' ],
               cmap=plt.cm.get_cmap('magma').reversed(),
               save=f'_{namespace}_origins.png')

    sc.pl.umap(adata, color='pseudotime', edges=True, cmap='inferno',
               edges_color='#aaaaaa', save=f'_{namespace}_pseudotime.png')

    nnan_idx = (np.isfinite(adata.obs['timestamp']) &
                np.isfinite(adata.obs['pseudotime']))

    adata_nnan = adata[nnan_idx]

    plt.figure()
    sns.regplot(x='timestamp', y='pseudotime', ci=None,
                data=adata_nnan.obs)
    plt.ylim([ -0.01, 1.01 ])
    plt.savefig(f'figures/{namespace}_pseudotime-time.png', dpi=500)
    plt.tight_layout()
    plt.close()

    with open(f'target/ev_cache/{namespace}_pseudotime.txt', 'w') as of:
        of.write('\n'.join([ str(x) for x in adata.obs['pseudotime'] ]) + '\n')

    tprint('Pseudotime-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata_nnan.obs['pseudotime'],
                                 adata_nnan.obs['timestamp'],
                                 nan_policy='omit')))
    tprint('Pseudotime-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata_nnan.obs['pseudotime'],
                                adata_nnan.obs['timestamp'])))

    seqlens = [ len(seq) for seq in adata.obs['seq'] ]
    tprint('Pseudotime-length Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudotime'], seqlens,
                                 nan_policy='omit')))


if __name__ == '__main__':
    args = parse_args()

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

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        spike_evolocity(args, model, seqs, vocabulary,
                        namespace=args.namespace)

        if args.model_name == 'esm1b' and args.velocity_score == 'lm':
            tprint('One hot featurization:')
            spike_evolocity(args, model, seqs, vocabulary, namespace='cov_onehot')
