from torchtext.vocab import GloVe, vocab 

def get_glove_vocab(opt):
    glove_vectors = GloVe(name=opt.vector_name, dim=opt.embedding_dim, cache=opt.cache_dir)
    unk_token = '<unk>'
    glove_vocab = vocab(glove_vectors.stoi, specials=[unk_token])
    glove_vocab.set_default_index(glove_vocab[unk_token])
    return glove_vocab, glove_vectors    

def get_vocab(opt):
    if opt.pretrain == 'glove':
        return get_glove_vocab(opt)
