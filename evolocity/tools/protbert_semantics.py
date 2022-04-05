import torch

def predict_sequence_prob_protbert(seq, model):
    tokenizer = model.tokenizer_

    seq = " ".join(seq)

    token_ids = torch.tensor([ tokenizer.encode(seq) ])
    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
    output = model.model_(token_ids)
    output = torch.nn.LogSoftmax(dim=2)(output[0])
    sequence_output = output.cpu().detach().numpy()

    return sequence_output[0]

def get_protbert_embedding(seq, tokenizer, model):
    encoded_input = tokenizer(seq, return_tensors='pt')
    output = model(**encoded_input)
    return output.logits.reshape((output.logits.shape[1], output.logits.shape[2])).detach().numpy()


def embed_seqs_protbert(
        seqs, model, tokenizer
):
    return {seq: [{'embedding': get_protbert_embedding(seq, tokenizer, model)}] for seq in seqs}


