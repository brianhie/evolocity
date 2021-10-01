from mutation import *
from evolocity_graph import *
import evolocity as evo

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
    parser.add_argument('--velocity-score', type=str, default='lm',
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

    import pickle
    cache_fname = 'target/ev_cache/np_seqs.pkl'
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

def plot_umap(adata, namespace='np'):
    sc.pl.umap(adata, color='year', save=f'_{namespace}_year.png',
               edges=True,)
    sc.pl.umap(adata, color='louvain', save=f'_{namespace}_louvain.png',
               edges=True,)
    sc.pl.umap(adata, color='subtype', save=f'_{namespace}_subtype.png',
               edges=True,)
    sc.pl.umap(adata, color='simple_subtype',
               save=f'_{namespace}_simple_subtype.png', edges=True,)
    sc.pl.umap(adata, color='host', save=f'_{namespace}_host.png',
               edges=True,)
    sc.pl.umap(adata, color='resist_adamantane',
               save=f'_{namespace}_adamantane.png', edges=True,)
    sc.pl.umap(adata, color='resist_oseltamivir',
               save=f'_{namespace}_oseltamivir.png', edges=True,)
    sc.pl.umap(adata, color='virulence', save=f'_{namespace}_virulence.png',
               edges=True,)
    sc.pl.umap(adata, color='transmission', save=f'_{namespace}_transmission.png',
               edges=True,)
    sc.pl.umap(adata, color='homology', save=f'_{namespace}_homology.png',
               edges=True,)

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

    ax.scatter(gong_x, gong_y, s=50, c=gong_c, cmap='Oranges',
               edgecolors='black', linewidths=0.5, zorder=10)

def analyze_edges(adata, model, vocabulary, namespace='np'):
    from evolocity.tools.velocity_graph import VelocityGraph

    vgraph = VelocityGraph(adata, adata.obs['seq'])
    n_obs = adata.X.shape[0]

    vgraph.compute_likelihoods(vocabulary, model)

    dirname = f'target/{namespace}'
    mkdir_p(dirname)
    with open(f'{dirname}/{namespace}_edges.txt', 'w') as of:
        for i in tqdm(range(n_obs)):
            if '_blosum' in namespace:
                score_fn = likelihood_blosum62
            else:
                score_fn = likelihood_muts
            neighs_idx = get_iterative_indices(
                vgraph.indices, i, vgraph.n_recurse_neighbors, vgraph.max_neighs
            )
            for j in neighs_idx:
                val = score_fn(
                    vgraph.seqs[i], vgraph.seqs[j],
                    args, vocabulary, model,
                    seq_cache=vgraph.seq_probs, verbose=vgraph.verbose,
                )
                fields = [
                    i, j, adata.obs['year'][i], adata.obs['year'][j], val
                ]
                of.write('\t'.join([ str(field) for field in fields ]) + '\n')

def epi_gong2013(args, model, seqs, vocabulary, namespace='np'):
    if args.model_name != 'esm1b':
        namespace += f'_{args.model_name}'
    if args.velocity_score != 'lm':
        namespace += f'_{args.velocity_score}'

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

    if namespace == 'np':
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
        gong_x = list(range(len(df) + 1))
        gong_y = [ 0 ] + list(np.cumsum(df['muts']))
        tprint('Sum of full scores: {}'.format(sum(df.full)))
        tprint('Sum of local scores: {}'.format(sum(df.muts)))
        tprint('Sum of self scores: {}'.format(sum(df.self_score)))
        tprint('Gong et al. Spearman r = {}, P = {}'
               .format(*ss.spearmanr(gong_x, gong_y)))

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

        sc.pp.neighbors(adata, n_neighbors=40, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

        adata.write(adata_cache)

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
        'H1N1', 'H2N2', 'H3N2', 'H5N1', 'H7N9',
    }
    adata.obs['simple_subtype'] = [
        subtype if subtype in keep_subtypes else 'other/unknown'
        for subtype in adata.obs['subtype']
    ]

    tprint('Analyzing {} sequences...'.format(adata.X.shape[0]))
    evo.set_figure_params(dpi_save=500, figsize=(5, 5))
    plot_umap(adata, namespace=namespace)

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
        evo.tl.velocity_graph(adata, model_name=args.model_name,
                              score=args.velocity_score)
        from scipy.sparse import save_npz
        save_npz('{}_vgraph.npz'.format(cache_prefix),
                 adata.uns["velocity_graph"],)
        save_npz('{}_vgraph_neg.npz'.format(cache_prefix),
                 adata.uns["velocity_graph_neg"],)
        np.save('{}_vself_transition.npy'.format(cache_prefix),
                adata.obs["velocity_self_transition"],)

    # Edge score stratification analysis.
    #analyze_edges(adata, model, vocabulary, namespace=namespace)

    rw_root = list(adata.obs['seq']).index(nodes[0][1])

    if namespace == 'np':
        evo.tl.random_walk(
            adata,
            root_node=rw_root,
            walk_length=len(nodes) - 1,
            n_walks=30000,
            groupby='subtype',
            groups='H3N2',
            scale=2.,
        )
        
        terminal_clusters = { '1', '3', '8', '9' }
        paths = adata.uns['rw_paths']
        
        plt.figure(figsize=(8, 3))
        plt.scatter(gong_x, gong_y, s=50, c=gong_x, cmap='Oranges',
                    edgecolors='black', linewidths=0.5, zorder=10)
        plt.plot(gong_x, gong_y, c='black', zorder=9)
        for p in range(paths.shape[0]):
            if adata.obs['louvain'][paths[p][-1]] in terminal_clusters:
                walk_v = []
                for idx, seq in enumerate(paths[p]):
                    if idx == 0:
                        walk_v.append(0)
                        continue
                    seq_prev = paths[p][idx - 1]
                    walk_v.append(adata.uns['velocity_graph'][seq_prev, seq])
                plt.plot(gong_x, np.cumsum(walk_v),
                         c='000080', alpha=0.1, zorder=5)
        plt.ylim([ -2, 14 ])
        plt.axhline(c='black', linestyle='--')
        plt.savefig(f'figures/{namespace}_gong_path.svg')
        plt.close()

        evo.tl.onehot_msa(
            adata,
            reference=list(adata.obs['gene_id']).index('H1N1_1934_human_>J02147'),
            dirname=f'target/evolocity_alignments/{namespace}',
            seq_id_fields=[ 'subtype', 'year' ],
            n_threads=40,
        )
        evo.tl.residue_scores(adata)
        evo.pl.residue_scores(
            adata,
            save=f'_{namespace}_residue_scores.png',
        )
        evo.pl.residue_categories(
            adata,
            namespace=namespace,
            n_plot=10,
            reference=list(adata.obs['gene_id']).index('H1N1_1934_human_>J02147'),
        )

    evo.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    evo.pl.velocity_embedding(
        adata, basis='umap', color='year', save=f'_{namespace}_year_velo.png',
    )

    # Grid visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=3., smooth=1.,
        arrow_size=1., arrow_length=3., alpha=0.9,
        color='year', show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/evolocity__{namespace}_year_velogrid.png', dpi=500)
    plt.close()

    if namespace == 'np':
        plt.figure()
        ax = evo.pl.velocity_embedding_grid(
            adata, basis='umap', min_mass=3., smooth=1.,
            arrow_size=1., arrow_length=3., alpha=1.,
            color='pos373', show=False,
        )
        sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, 'aaaaaa')
        plt.tight_layout(pad=1.1)
        plt.subplots_adjust(right=0.85)
        plt.savefig(f'figures/evolocity__{namespace}_pos373_velogrid.png', dpi=500)
        plt.close()

    # Streamplot visualization.
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=4., smooth=1., density=1.2,
        color='year', show=False, colorbar=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/evolocity__{namespace}_year_velostream.png', dpi=500)
    plt.close()
    plt.figure()
    ax = evo.pl.velocity_embedding_stream(
        adata, basis='umap', min_mass=4., smooth=1., density=1.2,
        color='simple_subtype', legend_loc=False, show=False,
    )
    sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, '#aaaaaa')
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/evolocity__{namespace}_subtype_velostream.png', dpi=500)
    plt.close()
    if namespace == 'np':
        plt.figure()
        ax = evo.pl.velocity_embedding_stream(
            adata, basis='umap', min_mass=4., smooth=1., density=1.2,
            color='pos104', show=False, legend_loc=False,
            palette=[ '888888', '1f77b4', '888888',
                      '888888', 'd62728', '888888',
                      '2ca02c', '9467bd', ],
        )
        sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, 'aaaaaa')
        plt.tight_layout(pad=1.1)
        plt.subplots_adjust(right=0.85)
        plt.savefig(f'figures/evolocity__{namespace}_pos104_velostream.png',
                    dpi=500)
        plt.close()
        plt.figure()
        ax = evo.pl.velocity_embedding_stream(
            adata, basis='umap', min_mass=4., smooth=1., density=1.2,
            color='pos238', show=False, legend_loc=False,
            palette=[ '888888', '1f77b4', 'd62728',
                      '2ca02c', '9467bd', ],
        )
        sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, 'aaaaaa')
        plt.tight_layout(pad=1.1)
        plt.subplots_adjust(right=0.85)
        plt.savefig(f'figures/evolocity__{namespace}_pos238_velostream.png',
                    dpi=500)
        plt.close()
        plt.figure()
        ax = evo.pl.velocity_embedding_stream(
            adata, basis='umap', min_mass=4., smooth=1., density=1.2,
            color='pos455', show=False, legend_loc=False,
            palette=[ '888888', '1f77b4', 'ff7f0e',
                      'd62728', '9467bd', ],
        )
        sc.pl._utils.plot_edges(ax, adata, 'umap', 0.1, 'aaaaaa')
        plt.tight_layout(pad=1.1)
        plt.subplots_adjust(right=0.85)
        plt.savefig(f'figures/evolocity__{namespace}_pos455_velostream.png',
                    dpi=500)
        plt.close()

    plt.figure()
    ax = evo.pl.velocity_contour(
        adata,
        basis='umap', smooth=1., pt_smooth=0.8, levels=100,
        density=0.6, arrow_size=4, arrow_length=3., cmap='coolwarm',
        c='#aaaaaa', show=False,
    )
    plt.tight_layout(pad=1.1)
    draw_gong_path(ax, adata)
    plt.savefig(f'figures/evolocity__{namespace}_contour.png', dpi=500)
    plt.close()

    sc.pl.umap(adata, color=[ 'root_nodes', 'end_points' ],
               color_map=plt.cm.get_cmap('magma').reversed(),
               edges=True, edges_color='#aaaaaa',
               save=f'_{namespace}_origins.png')

    sc.pl.umap(adata, color='pseudotime', edges=True, cmap='inferno',
               save=f'_{namespace}_pseudotime.png')

    nnan_idx = (np.isfinite(adata.obs['year']) &
                np.isfinite(adata.obs['pseudotime']))

    adata_nnan = adata[nnan_idx]

    with open(f'target/ev_cache/{namespace}_pseudotime.txt', 'w') as of:
        of.write('\n'.join([ str(x) for x in adata.obs['pseudotime'] ]) + '\n')

    tprint('Pseudotime-time Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata_nnan.obs['pseudotime'],
                                 adata_nnan.obs['year'],
                                 nan_policy='omit')))
    tprint('Pseudotime-time Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata_nnan.obs['pseudotime'],
                                adata_nnan.obs['year'])))

    tprint('Pseudotime-length Spearman r = {}, P = {}'
           .format(*ss.spearmanr(adata_nnan.obs['pseudotime'],
                                 adata_nnan.obs['seqlen'],
                                 nan_policy='omit')))
    tprint('Pseudotime-length Pearson r = {}, P = {}'
           .format(*ss.pearsonr(adata_nnan.obs['pseudotime'],
                                adata_nnan.obs['seqlen'])))

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
        tprint('NP analysis:')
        epi_gong2013(args, model, seqs, vocabulary, namespace=namespace)

        if args.model_name == 'esm1b' and args.velocity_score == 'lm':
            tprint('NP with one-hot features analysis:')
            epi_gong2013(args, model, seqs, vocabulary, namespace='np_onehot')
