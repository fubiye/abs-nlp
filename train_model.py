from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
from absnlp.vocab.pretrain_static import get_vocab
from absnlp.dataset.conll import preprocess_tags
from absnlp.model.rnn import SimpleRnnNet

import logging
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    for epoch in range(opt.epoches):
        logger.info("start epoch: %d", epoch)
        for batch, (sents, sent_tags) in enumerate(train_loader):
            logger.info('batch: %d, sents size: %d', batch, len(sents))
            sents = sents.to(device)
            tags_hat = model(sents)
            for i in range(len(sents)):
                print(sents[i])
                print(sent_tags[i])
                print(tags_hat)
                print('\n')
                
            if batch > 0:
                return
if __name__ == '__main__':

    opt = ParserInit().opt
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader = get_loaders(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt).to(device)
    train()
    