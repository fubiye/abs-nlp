from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
from absnlp.vocab.pretrain_static import get_vocab
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

if __name__ == '__main__':
    opt = ParserInit().opt
    vocab, vectors = get_vocab(opt)

    token = 'said'
    print(vocab[token], vectors[vocab[token]])
    token = '<unk>'
    print(vocab[token], vectors[vocab[token]])
    token = 'out of vocab'
    print(vocab[token], vectors[vocab[token]])
    # train_loader = get_loaders(opt)
