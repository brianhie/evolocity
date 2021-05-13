from scanpy import read

url_datadir = 'https://github.com/brianhie/evovdss/raw/master/'

def nucleoprotein():
    """Influenza A nucleoprotein.

    Data downloaded from the NIAID Influenza Research Database
    (https://www.fludb.org/).
    Most sequences include metadata on the sampling year and the
    influenza subtype.

    The sequence landscape shows structure according to both
    sampling year and subtype.

    Returns
    -------
    Returns `adata` object
    """

    fname = 'target/ev_cache/np_adata.h5ad'
    url = f'{url_datadir}np_adata.h5ad'
    adata = read(fname, backup_url=url, sparse=True, cache=True)

    return adata

def cytochrome_c():
    """Eukaryotic cytochrome c.

    Data downloaded from UniProt (https://www.uniprot.org/). Sequences
    are from the "cytochrome c" family and filtered to preserve the
    largest mode in sequence lengths and to preserve eukyarotic
    proteins.

    The sequence landscape capture the diversification of the eukaryota.

    Returns
    -------
    Returns `adata` object
    """

    fname = 'target/ev_cache/cyc_adata.h5ad'
    url = f'{url_datadir}cyc_adata.h5ad'
    adata = read(fname, backup_url=url, sparse=True, cache=True)

    return adata
