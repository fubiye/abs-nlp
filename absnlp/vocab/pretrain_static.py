from torchtext.vocab import GloVe, vocab 

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>','<pad>','<bos>','<eos>']

def get_glove_vocab(opt):
    glove_vectors = GloVe(name=opt.vector_name, dim=opt.embedding_dim, cache=opt.glove_cache)

    glove_vocab = vocab(glove_vectors.stoi, specials=special_symbols)
    glove_vocab.set_default_index(UNK_IDX)
    return glove_vocab, glove_vectors    

def get_vocab(opt):
    if opt.pretrain == 'glove':
        return get_glove_vocab(opt)
