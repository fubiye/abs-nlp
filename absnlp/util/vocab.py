from torchtext.vocab import GloVe, vocab
import torch 
UNK_TOKEN = "<unk>"
UNK_INDEX = 0


def load_glove(opt):
    glove = GloVe(name=opt.glove_name, dim=opt.embedding_size, cache=opt.glove_cache)
    glove_vocab = vocab(glove.stoi)
    glove_vocab.insert_token(UNK_TOKEN, UNK_INDEX)
    glove_vocab.set_default_index(UNK_INDEX)

    embeddings = glove.vectors
    embeddings = torch.cat((torch.zeros(1, embeddings.shape[1]),embeddings))

    return glove_vocab, embeddings