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
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []

    for epoch in range(opt.epoches):
        logger.info("start epoch: %d", epoch)
        total_loss = 0.
        for batch, (sents, target_tags) in enumerate(train_loader):
            batch_size = sents.shape[0]
            sents = sents.to(device)
            target_tags = target_tags.to(device)
            logits = model(sents)
            predicted = logits.permute(0,2,1)
            loss = loss_fn(predicted, target_tags)
            total_loss += loss.item()
            # print('[epoch: {} batch: {} loss: {}'.format(epoch, batch, loss.item()))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if (batch + 1) % 10 == 0:
                cur_loss = total_loss
                loss_history.append(cur_loss / (batch+1))
                # f1_history.append(f1 / (idx+1))
                total_loss = 0
                
                num_ele = batch * batch_size
                print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, batch, (cur_loss / num_ele), 'f1 placeholder'))
if __name__ == '__main__':

    opt = ParserInit().opt
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader = get_loaders(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    train()
    