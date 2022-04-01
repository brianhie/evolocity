from ..preprocessing import get_model

def velocity_model(
    adata,
    model_name='esm1b',
    mkey='model',
    copy=False,
    model_path=None
):
    adata = adata.copy() if copy else adata

    model = get_model(model_name, model_path=model_path)

    adata.uns[mkey] = model

    return adata if copy else None
