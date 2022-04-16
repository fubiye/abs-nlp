from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
from absnlp.vocab.pretrain_static import PAD_IDX, get_vocab
from absnlp.dataset.conll import preprocess_tags
from absnlp.model.rnn import SimpleRnnNet
import absnlp.util.figure as figure
import logging
import torch
import torchmetrics


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
            acc = acc_metric(predicted, target_tags)
            f1_score = f1_metric(predicted,target_tags)
            print('[epoch: {} batch: {} loss: {} acc: {} f1:{} '.format(epoch, batch, loss.item(), acc, f1_score))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if (batch + 1) % 10 == 0:
                cur_loss = total_loss
                loss_history.append(cur_loss)
                # f1_history.append(f1 / (idx+1))
                total_loss = 0
                
                num_ele = batch * batch_size
                print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, batch, (cur_loss), 'f1 placeholder'))
                iters = [i for i in range(len(loss_history))]
                figure.draw(iters, loss_history)
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        print(f"Accuracy on all data: {acc}\n F1 score: {f1}")
        acc_metric.reset()
        f1_metric.reset()

def eval():
    model.eval()
    
if __name__ == '__main__':

    opt = ParserInit().opt
    figure.init_plt()
    acc_metric = torchmetrics.Accuracy()
    f1_metric = torchmetrics.F1Score(num_classes=opt.num_of_tags, multiclass=True, mdmc_average='samplewise')
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader, test_loader = get_loaders(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    train()
    