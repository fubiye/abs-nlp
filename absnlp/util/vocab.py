from torchtext.vocab import GloVe, vocab
import torch 
UNK_TOKEN = '<unk>'

def load_glove(args):

    glove = GloVe(name=args.vector_name, dim=args.embedding_dim, cache=args.glove_cache)
    glove_vocab = vocab(glove.stoi)
    glove_vocab.insert_token(UNK_TOKEN, args.default_index)
    glove_vocab.set_default_index(args.default_index)

    embeddings = glove.vectors
    embeddings = torch.cat((torch.zeros(1, embeddings.shape[1]),embeddings))

    return glove_vocab, embeddings