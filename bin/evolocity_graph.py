from scvelo.preprocessing.neighbors import neighbors, verify_neighbors
from scvelo.preprocessing.neighbors import get_neighs, get_n_neighs

from scipy.sparse import coo_matrix, issparse

from utils import *

def norm(A):
    """computes the L2-norm along axis 1
    (e.g. genes or embedding dimensions) equivalent to np.linalg.norm(A, axis=1)
    """
    if issparse(A):
        return np.sqrt(A.multiply(A).sum(1).A1)
    else:
        return np.sqrt(np.einsum("ij, ij -> i", A, A)
                       if A.ndim > 1 else np.sum(A * A))

def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):
    from scvelo.preprocessing.neighbors import compute_connectivities_umap

    D = dist.copy()
    D.data += 1e-6

    n_counts = sum_var(D > 0)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D

class VelocityGraph:
    def __init__(
        self,
        adata,
        vkey="velocity",
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        mode_neighbors="distances",
    ):
        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
            raise ValueError(
                "Your neighbor graph seems to be corrupted. "
                "Consider recomputing via scanpy.pp.neighbors."
            )
        self.indices = get_indices(
            dist=get_neighs(adata, "distances"),
            mode_neighbors=mode_neighbors,
        )[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        self.self_prob = None
        self.adata = adata

    def compute_gradients(self):
        vals, rows, cols, uncertainties, n_obs = [], [], [], [], self.X.shape[0]

        for i in range(n_obs):
            neighs_idx = get_iterative_indices(
                self.indices, i, self.n_recurse_neighbors, self.max_neighs
            )

            # Compute probabilities here.
            val = np.zero(np.zeros(len(neighs_idx)))

            vals.extend(val)
            rows.extend(np.ones(len(neighs_idx)) * i)
            cols.extend(neighs_idx)

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)


def evelocity_graph(
    data,
    vkey="velocity",
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    mode_neighbors="distances",
    copy=False,
):
    adata = data.copy() if copy else data
    verify_neighbors(adata)

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        mode_neighbors=mode_neighbors,
    )

    if verbose:
        tprint("Computing velocity graph...")
    vgraph.compute_gradients()

    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg
    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    return adata if copy else None
