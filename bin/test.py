from flu_np import *

def setup(args):
    fnames = [ 'data/influenza/ird_influenzaA_NP_allspecies.fa' ]
    meta_fnames = fnames

    seqs = process(args, fnames, meta_fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size,
                      inference_batch_size=1000)

    return model, seqs

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

def test(args, model, seqs, vocabulary):
    nodes = [
        (record.id, str(record.seq))
        for record in SeqIO.parse('data/influenza/np_nodes.fa', 'fasta')
    ]

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
    sc.tl.umap(adata, min_dist=1.)

    sc.set_figure_params(dpi_save=500)
    sc.tl.louvain(adata, resolution=1.)

    sc.set_figure_params(dpi_save=500)
    #plot_umap(adata)
    #sc.pl.umap(adata, color='gong2013_step', save='_np_gong2013.png',
    #           edges=True,)

    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
    from scipy.sparse import load_npz
    adata.uns["velocity_graph"] = load_npz('target/np_knn40_vgraph.npz')
    adata.uns["velocity_graph_neg"] = load_npz(
        'target/np_knn40_vgraph_neg.npz'
    )
    adata.obs["velocity_self_transition"] = np.load(
        'target/np_knn40_vself_transition.npy'
    )
    adata.layers["velocity"] = np.zeros(adata.X.shape)

    import scvelo as scv
    scv.tl.velocity_embedding(adata, basis='umap', scale=1.,
                              self_transitions=True,
                              use_negative_cosines=True,
                              retain_scale=False,
                              autoscale=True,)
    scv.pl.velocity_embedding(
        adata, basis='umap', color='year', save='_np_year_velo.png',
    )

    plt.figure()
    ax = scv.pl.velocity_embedding_grid(
        adata, basis='umap', min_mass=4., smooth=1.2,
        arrow_size=1., arrow_length=3.,
        color='year', show=False,
    )
    plt.tight_layout(pad=1.1)
    plt.subplots_adjust(right=0.85)
    draw_gong_path(ax, adata)
    plt.savefig('figures/scvelo__np_year_velogrid.png', dpi=500)
    plt.close()

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
    plt.savefig('figures/scvelo__np_year_velostream.png', dpi=500)
    plt.close()

    scv.tl.terminal_states(adata)
    scv.pl.scatter(adata, color=[ 'root_cells', 'end_points' ],
                   cmap=plt.cm.get_cmap('magma').reversed(),
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

if __name__ == '__main__':
    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]

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

    test(args, model, seqs, vocabulary)
