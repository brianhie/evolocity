from mutation import *
from evolocity_graph import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Flu NP sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='np',
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
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--epistasis', action='store_true',
                        help='Analyze epistasis')
    args = parser.parse_args()
    return args

def parse_phenotype(field):
    field = field.split('_')[-1]
    if field == 'No':
        return 'no'
    elif field == 'Yes':
        return 'yes'
    else:
        return 'unknown'

def load_meta(meta_fnames):
    with open('data/influenza/np_birds.txt') as f:
        birds = set(f.read().lower().rstrip().split())
    with open('data/influenza/np_mammals.txt') as f:
        mammals = set(f.read().lower().rstrip().split())

    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            for line in f:
                if not line.startswith('>'):
                    continue
                accession = line[1:].rstrip()
                fields = line.rstrip().split('|')

                subtype = fields[4]
                year = fields[5]
                date = fields[5]
                country = fields[7]
                host = fields[9].lower()
                resist_adamantane = parse_phenotype(fields[12])
                resist_oseltamivir = parse_phenotype(fields[13])
                virulence = parse_phenotype(fields[14])
                transmission = parse_phenotype(fields[15])

                if year == '-' or year == 'NA' or year == '':
                    year = None
                else:
                    year = int(year.split('/')[-1])

                if date == '-' or date == 'NA' or date == '':
                    date = None
                else:
                    date = dparse(date)

                if host in birds:
                    host = 'avian'
                elif host in mammals:
                    host = 'other_mammal'

                metas[accession] = {
                    'subtype': subtype,
                    'year': year,
                    'date': date,
                    'country': country,
                    'host': host,
                    'resist_adamantane': resist_adamantane,
                    'resist_oseltamivir': resist_oseltamivir,
                    'virulence': virulence,
                    'transmission': transmission,
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
            if meta['seqlen'] < 450:
                continue
            if 'X' in record.seq:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            seqs[record.seq].append(meta)

    tprint('Found {} unique sequences'.format(len(seqs)))

    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs = {}, {}

    old_cutoff = 1900
    new_cutoff = 2008

    tprint('Splitting seqs...')
    for seq in seqs:
        # Pick validation set based on date.
        seq_dates = [
            meta['year'] for meta in seqs[seq]
            if meta['year'] is not None
        ]
        if len(seq_dates) == 0:
            test_seqs[seq] = seqs[seq]
            continue
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
    fnames = [ 'data/influenza/ird_influenzaA_NP_allspecies.fa' ]
    meta_fnames = fnames

    seqs = process(args, fnames, meta_fnames)

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
    sc.tl.umap(adata, min_dist=1.)
    sc.pl.umap(adata, color='louvain', save='_np_louvain.png')
    sc.pl.umap(adata, color='subtype', save='_np_subtype.png')
    sc.pl.umap(adata, color='year', save='_np_year.png',
               edges=True,)
    sc.pl.umap(adata, color='host', save='_np_host.png')
    sc.pl.umap(adata, color='resist_adamantane', save='_np_adamantane.png')
    sc.pl.umap(adata, color='resist_oseltamivir', save='_np_oseltamivir.png')
    sc.pl.umap(adata, color='virulence', save='_np_virulence.png')
    sc.pl.umap(adata, color='transmission', save='_np_transmission.png')

def populate_embedding(args, model, seqs, vocabulary,
                       use_cache=False, namespace=None):
    if namespace is None:
        namespace = args.namespace

    if use_cache:
        mkdir_p('target/{}/embedding'.format(namespace))
        embed_prefix = ('target/{}/embedding/{}_{}'
                        .format(namespace, args.model_name, args.dim))

    sorted_seqs = np.array([ str(s) for s in sorted(seqs.keys()) ])
    batch_size = 3000
    n_batches = math.ceil(len(sorted_seqs) / float(batch_size))
    for batchi in range(n_batches):
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
        seqs_batch = embed_seqs(args, model, seqs_batch, vocabulary,
                                use_cache=False)
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

    #adata = adata[
    #    np.logical_or.reduce((
    #        adata.obs['Host Species'] == 'human',
    #        adata.obs['Host Species'] == 'avian',
    #        adata.obs['Host Species'] == 'swine',
    #    ))
    #]

    sc.pp.neighbors(adata, n_neighbors=200, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata)

    interpret_clusters(adata)

def draw_gong_path(ax, adata):
    gong_adata = adata[adata.obs['gong2013_step'].astype(float) > 0]
    gong_sort_idx = np.argsort(gong_adata.obs['gong2013_step'])
    gong_c = gong_adata.obs['gong2013_step'][gong_sort_idx]
    gong_x = gong_adata.obsm['X_umap'][gong_sort_idx, 0]
    gong_y = gong_adata.obsm['X_umap'][gong_sort_idx, 1]

    for idx, (x, y) in enumerate(zip(gong_x, gong_y)):
        if idx < len(gong_x) - 1:
            dx, dy = gong_x[idx + 1] - x, gong_y[idx + 1] - y
            ax.arrow(x, y, dx, dy, width=0.001, head_width=0.,
                     length_includes_head=True,
                     color='#888888', zorder=5)

    ax.scatter(gong_x, gong_y, s=15, c=gong_c, cmap='Greys',
               edgecolors='black', linewidths=0.5, zorder=10)

def epi_gong2013(args, model, seqs, vocabulary):
    ###############
    ## Load data ##
    ###############

    nodes = [
        (record.id, str(record.seq))
        for record in SeqIO.parse('data/influenza/np_nodes.fa', 'fasta')
    ]

    ######################################
    ## See how local likelihoods change ##
    ######################################

    data = []
    for idx, (name, seq) in enumerate(nodes):
        if idx > 0:
            seq_prev = nodes[idx - 1][1]
            score_full = likelihood_full(seq_prev, seq,
                                         args, vocabulary, model,)
            score_muts = likelihood_muts(seq_prev, seq,
                                         args, vocabulary, model,)
            score_self = likelihood_self(seq_prev, seq,
                                         args, vocabulary, model,)
            data.append([ name, seq,
                          score_full, score_muts, score_self ])
            tprint('{}: {}, {}, {}'.format(name, score_full,
                                           score_muts, score_self))

    df = pd.DataFrame(data, columns=[ 'name', 'seq', 'full', 'muts',
                                      'self_score' ])
    tprint('Sum of full scores: {}'.format(sum(df.full)))
    tprint('Sum of local scores: {}'.format(sum(df.muts)))
    tprint('Sum of self scores: {}'.format(sum(df.self_score)))

    ############################
    ## Visualize NP landscape ##
    ############################

    seqs = populate_embedding(args, model, seqs, vocabulary,
                              use_cache=True)

    for seq in seqs:
        for example_meta in seqs[seq]:
            example_meta['gong2013_step'] = 0
    for node_idx, (_, seq) in enumerate(nodes):
        if seq in seqs:
            for meta in seqs[seq]:
                meta['gong2013_step'] = node_idx + 100
        else:
            meta = {}
            for key in example_meta:
                meta[key] = None
            meta['embedding'] = embed_seqs(
                args, model, { seq: [ {} ] }, vocabulary, verbose=False,
            )[seq][0]['embedding'].mean(0)
            meta['gong2013_step'] = node_idx + 100
            seqs[seq] = [ meta ]

    adata = seqs_to_anndata(seqs)

    adata = adata[(adata.obs.host == 'human')]

    sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')

    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    plot_umap(adata)
    sc.pl.umap(adata, color='gong2013_step', save='_np_gong2013.png',
               edges=True,)

    #####################################
    ## Compute evolocity and visualize ##
    #####################################

    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
    velocity_graph(adata, args, vocabulary, model,
                   n_recurse_neighbors=0,)

    import scvelo as scv
    scv.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    scv.pl.velocity_embedding(
        adata, basis='umap', color='year', save='_np_year_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=4., smooth=1.,
        arrow_size=1., arrow_length=3.,
        color='year', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig('figures/scvelo__np_year_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=4., smooth=1.,
        color='year', show=False,
    )
    sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig('figures/scvelo__np_year_velostream.png', dpi=500)
    plt.close()

    scv.tl.terminal_states(adata)
    scv.pl.scatter(adata, color=['root_cells', 'end_points'],
                   save='_np_origins.png', dpi=500)
    nnan_idx = (np.isfinite(adata.obs['year']) &
                np.isfinite(adata.obs['root_cells']) &
                np.isfinite(adata.obs['end_points']))
    tprint('Root-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['root_cells'][nnan_idx],
                                 adata.obs['year'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Root-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['root_cells'][nnan_idx],
                                adata.obs['year'][nnan_idx])))
    tprint('End-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['end_points'][nnan_idx],
                                 adata.obs['year'][nnan_idx],
                                 nan_policy='omit')))
    tprint('End-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['end_points'][nnan_idx],
                                adata.obs['year'][nnan_idx])))

    exit()

    from scipy.sparse import save_npz
    save_npz('target/np_knn40_vgraph.npz',
             adata.uns["velocity_graph"],)
    save_npz('target/np_knn40_vgraph_neg.npz',
             adata.uns["velocity_graph_neg"],)
    np.save('target/np_knn40_vself_transition.npy',
            adata.obs["velocity_self_transition"],)

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
                       if '<' not in tok }
        args.checkpoint = args.model_name
    elif args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.train:
        batch_train(args, model, seqs, vocabulary, batch_size=5000)

    if args.train_split or args.test:
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

    if args.epistasis:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        epi_gong2013(args, model, seqs, vocabulary)
