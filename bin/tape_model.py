import torch
from tape import ProteinBertForMaskedLM, TAPETokenizer

class TAPEModel(object):
    def __init__(self, name):
        self.name_ = name

        model = ProteinBertForMaskedLM.from_pretrained(name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model

        self.tokenizer_ = TAPETokenizer(vocab='iupac')

        self.alphabet_ = self.tokenizer_.vocab
