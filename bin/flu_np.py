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
    parser.add_argument('--evolocity', action='store_true',
                        help='Analyze evolocity')
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

                embl_id = fields[0]
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
                    'gene_id': f'{subtype}_{year}_{host}_{embl_id}',
                    'embl_id': embl_id,
                    'subtype': subtype,
                    'year': year,
                    'date': str(date),
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

    seqs = training_distances(seqs, namespace=args.namespace)

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

def plot_umap(adata, namespace='np'):
    sc.pl.umap(adata, color='louvain', save=f'_{namespace}_louvain.png')
    sc.pl.umap(adata, color='subtype', save=f'_{namespace}_subtype.png')
    sc.pl.umap(adata, color='year', save=f'_{namespace}_year.png',
               edges=True,)
    sc.pl.umap(adata, color='host', save=f'_{namespace}_host.png')
    sc.pl.umap(adata, color='resist_adamantane',
               save=f'_{namespace}_adamantane.png')
    sc.pl.umap(adata, color='resist_oseltamivir',
               save=f'_{namespace}_oseltamivir.png')
    sc.pl.umap(adata, color='virulence', save=f'_{namespace}_virulence.png')
    sc.pl.umap(adata, color='transmission', save=f'_{namespace}_transmission.png')
    sc.pl.umap(adata, color='homology', save=f'_{namespace}_homology.png')

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

    ax.scatter(gong_x, gong_y, s=15, c=gong_c, cmap='Oranges',
               edgecolors='black', linewidths=0.5, zorder=10)

def epi_gong2013(args, model, seqs, vocabulary, namespace='np'):

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
            tprint('{}: {}, {}, {}'.format(
                name, score_full, score_muts, score_self
            ))

    df = pd.DataFrame(data, columns=[ 'name', 'seq', 'full', 'muts',
                                      'self_score' ])
    tprint('Sum of full scores: {}'.format(sum(df.full)))
    tprint('Sum of local scores: {}'.format(sum(df.muts)))
    tprint('Sum of self scores: {}'.format(sum(df.self_score)))

    ############################
    ## Visualize NP landscape ##
    ############################

    adata_cache = 'target/ev_cache/np_adata.h5ad'
    try:
        import anndata
        adata = anndata.read_h5ad(adata_cache)
    except:
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
        adata.write(adata_cache)

    sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')

    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    sc.tl.umap(adata, min_dist=1.)
    plot_umap(adata, namespace=namespace)
    sc.pl.umap(adata, color='gong2013_step', save=f'_{namespace}_gong2013.png',
               edges=True,)

    #####################################
    ## Compute evolocity and visualize ##
    #####################################

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
        velocity_graph(adata, args, vocabulary, model)
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
        adata, basis='umap', color='year', save=f'_{namespace}_year_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=4., smooth=1.2,
        arrow_size=1., arrow_length=3.,
        color='year', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/scvelo__{namespace}_year_velogrid.png', dpi=500)
    plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = scv.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=4., smooth=1., density=1.2,
        color='year', show=False,
    )
    sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/scvelo__{namespace}_year_velostream.png', dpi=500)
    plt.close()


    plt.figure()
    ax = plot_pseudofitness(
        adata,
        basis='umap', smooth=1., pf_smooth=1.5, levels=100,
        arrow_size=1., arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
        rank_transform=True, use_ends=False,
    )
    plt.tight_layout(pad=1.1)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/scvelo__{namespace}_pseudofitness.png', dpi=500)
    plt.close()

    scv.pl.scatter(adata, color=[ 'root_cells', 'end_points' ],
                   cmap=plt.cm.get_cmap('magma').reversed(),
                   save=f'_{namespace}_origins.png', dpi=500)

    sc.pl.umap(adata, color='pseudofitness', edges=True, cmap='magma',
               save=f'_{namespace}_pseudofitness.png')

    nnan_idx = (np.isfinite(adata.obs['year']) &
                np.isfinite(adata.obs['pseudofitness']))
    tprint('Pseudofitness-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudofitness'][nnan_idx],
                                 adata.obs['year'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudofitness-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudofitness'][nnan_idx],
                                adata.obs['year'][nnan_idx])))

    nnan_idx = (np.isfinite(adata.obs['homology']) &
                np.isfinite(adata.obs['pseudofitness']))
    tprint('Pseudofitness-homology Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata.obs['pseudofitness'][nnan_idx],
                                 adata.obs['homology'][nnan_idx],
                                 nan_policy='omit')))
    tprint('Pseudofitness-homology Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata.obs['pseudofitness'][nnan_idx],
                                adata.obs['homology'][nnan_idx])))

    adata.write(f'target/results/{namespace}_adata.h5ad')

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
    elif args.model_name == 'tape':
        vocabulary = { tok: model.alphabet_[tok]
                       for tok in model.alphabet_ if '<' not in tok }
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

    if args.evolocity:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        namespace = args.namespace
        if args.model_name == 'tape':
            namespace += '_tape'
        epi_gong2013(args, model, seqs, vocabulary, namespace=namespace)
