from ..tools.velocity_pseudotime import velocity_pseudotime
from ..tools.velocity_embedding import velocity_embedding
from .utils import (
    default_arrow,
    default_basis,
    default_color,
    get_ax,
    get_basis,
    get_components,
    groups_to_bool,
    make_unique_list,
    savefig_or_show,
    velocity_embedding_changed,
)
from .velocity_embedding_grid import compute_velocity_on_grid

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def velocity_contour(
        adata,
        ptkey='pseudotime',
        rank_transform=True,
        use_ends=False,
        fill=True,
        levels=10,
        basis=None,
        vkey='velocity',
        density=None,
        smooth=None,
        pt_smooth=None,
        min_mass=None,
        arrow_size=None,
        arrow_length=None,
        arrow_color=None,
        scale=None,
        autoscale=True,
        n_neighbors=None,
        recompute=None,
        X=None,
        V=None,
        X_grid=None,
        V_grid=None,
        PF_grid=None,
        color=None,
        layer=None,
        color_map=None,
        colorbar=True,
        palette=None,
        size=None,
        alpha=0.5,
        offset=1,
        vmin=None,
        vmax=None,
        perc=None,
        sort_order=True,
        groups=None,
        components=None,
        projection='2d',
        legend_loc='none',
        legend_fontsize=None,
        legend_fontweight=None,
        xlabel=None,
        ylabel=None,
        title=None,
        fontsize=None,
        figsize=None,
        dpi=None,
        frameon=None,
        show=None,
        save=None,
        ax=None,
        ncols=None,
        **kwargs,
):
    """Contour plot of pseudotime with velocity grid.

    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    ptkey: `str` (default: `pseudotime`)
        Name of pseudotime values.
    rank_transform: `bool` (default: `True`)
        Perform final rank transformation.
    use_ends: `bool` (default: `False`)
        Use end terminal nodes in pseudotime computation.
    levels: `int` (default: 10)
        Number of contour levels.
    pt_smooth: `float` (default: `None`)
        Pseudotime two-dimensional smoothing.
    density: `float` (default: 1)
        Amount of velocities to show - 0 none to 1 all
    arrow_size: `float` or triple `headlength, headwidth, headaxislength` (default: 1)
        Size of arrows.
    arrow_length: `float` (default: 1)
        Length of arrows.
    scale: `float` (default: 1)
        Length of velocities in the embedding.
    min_mass: `float` or `None` (default: `None`)
        Minimum threshold for mass to be shown.
        It can range between 0 (all velocities) and 100 (large velocities).
    smooth: `float` (default: 0.5)
        Multiplication factor for scale in Gaussian kernel around grid point.
    n_neighbors: `int` (default: None)
        Number of neighbors to consider around grid point.
    X: `np.ndarray` (default: None)
        embedding grid point coordinates
    V: `np.ndarray` (default: None)
        embedding grid velocity coordinates
    {scatter}

    Returns
    -------
        `matplotlib.Axis` if `show==False`
    """
    if ptkey not in adata.obs:
        velocity_pseudotime(
            adata,
            vkey=vkey,
            groups=groups,
            rank_transform=rank_transform,
            use_velocity_graph=True,
            use_ends=use_ends,
        )
        adata.obs[ptkey] = adata.obs[f'{vkey}_pseudotime']

    smooth = 0.5 if smooth is None else smooth
    pt_smooth = smooth if pt_smooth is None else pt_smooth

    basis = default_basis(adata, **kwargs) \
            if basis is None \
            else get_basis(adata, basis)
    if vkey == 'all':
        lkeys = list(adata.layers.keys())
        vkey = [key for key in lkeys if 'velocity' in key and '_u' not in key]
    color, color_map = kwargs.pop('c', color), kwargs.pop('cmap', color_map)
    colors = make_unique_list(color, allow_array=True)
    layers, vkeys = (make_unique_list(layer),
                     make_unique_list(vkey))

    if V is None:
        for key in vkeys:
            if recompute or velocity_embedding_changed(
                    adata, basis=basis, vkey=key
            ):
                velocity_embedding(adata, basis=basis, vkey=key)

    color, layer, vkey = colors[0], layers[0], vkeys[0]
    color = default_color(adata) if color is None else color

    _adata = (
        adata[groups_to_bool(adata, groups, groupby=color)]
        if groups is not None and color in adata.obs.keys()
        else adata
    )
    comps, obsm = get_components(components, basis), _adata.obsm
    X_emb = np.array(obsm[f'X_{basis}'][:, comps]) \
            if X is None else X[:, :2]
    V_emb = np.array(obsm[f'{vkey}_{basis}'][:, comps]) \
            if V is None else V[:, :2]
    if X_grid is None or V_grid is None:
        X_grid, V_grid = compute_velocity_on_grid(
            X_emb=X_emb,
            V_emb=V_emb,
            density=density,
            autoscale=autoscale,
            smooth=smooth,
            n_neighbors=n_neighbors,
            min_mass=min_mass,
        )

    if vmin is None:
        vmin = adata.obs[ptkey].min()

    contour_kwargs = {
        'levels': levels,
        'vmin': vmin,
        'vmax': vmax,
        'alpha': alpha,
        'legend_fontsize': legend_fontsize,
        'legend_fontweight': legend_fontweight,
        'palette': palette,
        'cmap': color_map,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'colorbar': colorbar,
        'dpi': dpi,
    }

    ax, show = get_ax(ax, show, figsize, dpi)
    hl, hw, hal = default_arrow(arrow_size)
    if arrow_length is not None:
        scale = 1 / arrow_length
    if scale is None:
        scale = 1
    if arrow_color is None:
        arrow_color = 'grey'
    quiver_kwargs = {'angles': 'xy', 'scale_units': 'xy', 'edgecolors': 'k'}
    quiver_kwargs.update({'scale': scale, 'width': 0.001, 'headlength': hl / 2})
    quiver_kwargs.update({'headwidth': hw / 2, 'headaxislength': hal / 2})
    quiver_kwargs.update({'color': arrow_color, 'linewidth': 0.2, 'zorder': 3})

    for arg in list(kwargs):
        if arg in quiver_kwargs:
            quiver_kwargs.update({arg: kwargs[arg]})
        else:
            contour_kwargs.update({arg: kwargs[arg]})

    ax.quiver(
        X_grid[:, 0], X_grid[:, 1], V_grid[:, 0], V_grid[:, 1], **quiver_kwargs
    )

    PF_emb = np.array(adata.obs[ptkey]).reshape(-1, 1)
    if offset is not None:
        PF_emb += offset

    if PF_grid is None:
        _, mesh, PF_grid = compute_velocity_on_grid(
            X_emb=X_emb,
            V_emb=PF_emb,
            density=density,
            autoscale=False,
            smooth=pt_smooth,
            n_neighbors=n_neighbors,
            min_mass=0.,
            return_mesh=True,
        )
        PF_grid = PF_grid.reshape(mesh[0].shape)

    if fill:
        contour_fn = ax.contourf
    else:
        contour_fn = ax.contour

    contour = contour_fn(mesh[0], mesh[1], PF_grid, zorder=1,
                         **contour_kwargs)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #cbar = plt.colorbar(contour)
    #cbar.ax.set_ylabel(ptkey)

    savefig_or_show(dpi=dpi, save=save, show=show)
    if show is False:
        return ax
