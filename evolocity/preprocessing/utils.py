import os
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from anndata import AnnData
import warnings

from .. import logging as logg


def sum_obs(A):
    """summation over axis 0 (obs) equivalent to np.sum(A, 0)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.sum(0).A1 if issparse(A) else np.sum(A, axis=0)


def sum_var(A):
    """summation over axis 1 (var) equivalent to np.sum(A, 1)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.sum(1).A1 if issparse(A) else np.sum(A, axis=1)


def verify_dtypes(adata):
    try:
        _ = adata[:, 0]
    except:
        uns = adata.uns
        adata.uns = {}
        try:
            _ = adata[:, 0]
            logg.warn(
                "Safely deleted unstructured annotations (adata.uns), \n"
                "as these do not comply with permissible anndata datatypes."
            )
        except:
            logg.warn(
                "The data might be corrupted. Please verify all annotation datatypes."
            )
            adata.uns = uns


def get_size(adata, layer=None):
    X = adata.X if layer is None else adata.layers[layer]
    return sum_var(X)


def get_initial_size(adata, layer=None):
    if layer in adata.layers.keys():
        return (
            np.array(adata.obs[f"initial_size_{layer}"])
            if f"initial_size_{layer}" in adata.obs.keys()
            else get_size(adata, layer)
        )
    elif layer is None or layer == "X":
        return (
            np.array(adata.obs["initial_size"])
            if "initial_size" in adata.obs.keys()
            else get_size(adata)
        )
    else:
        return None


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
