import torch
from tape import ProteinBertForMaskedLM, TAPETokenizer

class TAPEModel(object):
    def __init__(self, name):
        self.name_ = 'tape'
        self.tape_name_ = name

        model = ProteinBertForMaskedLM.from_pretrained(name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model

        self.tokenizer_ = TAPETokenizer(vocab='iupac')
        self.alphabet_ = self.tokenizer_.vocab
        self.alphabet_['J'] = self.alphabet_['<unk>']
        self.unk_idx_ = self.tokenizer_.vocab['<unk>']

        self.vocabulary_ = {
            tok: self.alphabet_[tok]
            for tok in self.alphabet_ if '<' not in tok
        }
