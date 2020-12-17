import torch
import esm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

if __name__ == '__main__':
    name = 'esm1_t34_670M_UR50S'
    #name = 'esm1b_t33_650M_UR50S'

    model, alphabet = pretrained.load_model_and_alphabet(name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    assert(all(
        -(model.num_layers + 1) <= i <= model.num_layers
        for i in [ -1 ]
    ))
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1)
        for i in [ -1 ]
    ]

    seqs = [
        'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'MKTVRSERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'MKTVR-ERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
        'MKTVR0ERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
    ]
    labels = [
        'seq6Q',
        'seq6S',
        'seq6unk',
        'seq6mask'
    ]

    dataset = FastaBatchedDataset(labels, seqs)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches
    )

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            lsoftmax = torch.nn.LogSoftmax(dim=1)
            logits = lsoftmax(out["logits"]).to(device="cpu")

    print(logits.shape)

    print(logits[0, 6, alphabet.tok_to_idx['Q']])
    print(logits[0, 6, alphabet.tok_to_idx['S']])

    print(logits[1, 6, alphabet.tok_to_idx['Q']])
    print(logits[1, 6, alphabet.tok_to_idx['S']])

    print(logits[2, 6, alphabet.tok_to_idx['Q']])
    print(logits[2, 6, alphabet.tok_to_idx['S']])

    print(logits[3, 6, alphabet.tok_to_idx['Q']])
    print(logits[3, 6, alphabet.tok_to_idx['S']])
