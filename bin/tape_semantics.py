from utils import *

import torch

def predict_sequence_prob_tape(seq, model):
    tokenizer = model.tokenizer_

    token_ids = torch.tensor([ tokenizer.encode(seq) ])
    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
    output = model.model_(token_ids)
    output = torch.nn.LogSoftmax(dim=2)(output[0])
    sequence_output = output.cpu().detach().numpy()

    return sequence_output[0]
