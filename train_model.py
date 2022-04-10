from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
from absnlp.vocab.pretrain_static import PAD_IDX, get_vocab
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
        for batch, (sents, target_tags) in enumerate(train_loader):
            sents = sents.to(device)
            target_tags = target_tags.to(device)

            logits = model(sents)
            loss = loss_fn(logits.float(), target_tags)
            print('[epoch: {} batch: {} loss: {}'.format(epoch, batch, loss))
            # if batch > 0:
            #     return
if __name__ == '__main__':

    opt = ParserInit().opt
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader = get_loaders(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # loss_fn = torch.nn.NLLLoss(ignore_index=PAD_IDX)
    train()
    