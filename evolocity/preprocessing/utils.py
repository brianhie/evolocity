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


def check_if_valid_dtype(adata, layer="X"):
    X = adata.X if layer == "X" else adata.layers[layer]
    if "int" in X.dtype.name:
        if layer == "X":
            adata.X = adata.X.astype(np.float32)
        elif layer in adata.layers.keys():
            adata.layers[layer] = adata.layers[layer].astype(np.float32)
