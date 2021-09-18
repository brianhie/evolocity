import torch
from esm import pretrained, ProteinBertModel

class FBModel(object):
    def __init__(self, name, repr_layer=[-1], random_init=False):
        self.name_ = name
        self.repr_layer_ = repr_layer

        model, alphabet = pretrained.load_model_and_alphabet(name)

        if random_init:
            # ESM-1b with random initialization, for computational control.
            model = ProteinBertModel(
                args=model.args,
                alphabet=alphabet,
            )
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model
        self.alphabet_ = alphabet
        self.unk_idx_ = alphabet.tok_to_idx['<unk>']

        assert(all(
            -(model.num_layers + 1) <= i <= model.num_layers
            for i in [ -1 ]
        ))
        self.repr_layers_ = [
            (i + model.num_layers + 1) % (model.num_layers + 1)
            for i in [ -1 ]
        ]
