from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
from absnlp.vocab.pretrain_static import PAD_IDX, get_vocab
from absnlp.dataset.conll import preprocess_tags
from absnlp.model.rnn import SimpleRnnNet
import absnlp.util.figure as figure
import logging
import torch
import torchmetrics
import numpy as np
import os 

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    f1_history = []
    best_metric = 0
    for epoch in range(opt.epoches):
        model.train()
        logger.info("start epoch: %d", epoch)
        total_loss = 0.
        train_loss = []
        for batch, (sents, target_tags) in enumerate(train_loader):
            batch_size = sents.shape[0]
            sents = sents.to(device)
            target_tags = target_tags.to(device)
            logits = model(sents)
            predicted = logits.permute(0,2,1)
            loss = loss_fn(predicted, target_tags)
            total_loss += loss.item()
            train_loss.append(loss.item())
            acc = acc_metric(predicted, target_tags)
            f1_score = f1_metric(predicted,target_tags)
            loss.backward()
            optimizer.step()

            if (batch + 1) % 10 == 0:
                cur_loss = total_loss
                f1 = f1_metric.compute()
                loss_history.append(cur_loss)
                f1_history.append(f1)
                total_loss = 0
                num_ele = batch * batch_size
                print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch+1, batch, (cur_loss), f1))
                iters = [i for i in range(len(loss_history))]
                # figure.draw(iters, loss_history, f1_history)
                # break
        avg_loss =  np.average(train_loss)
        acc = acc_metric.compute()
        f1 = f1_metric.compute()
        logger.info("*"*10 + f": train result with epoch: {epoch}")
        logger.info(f"loss: {avg_loss}")
        logger.info(f"Accuracy:{acc}")
        logger.info(f"F1 Score: {f1}")
        acc_metric.reset()
        f1_metric.reset()
        logger.info("*"*10)
        _, eval_f1 = eval()
        if eval_f1 > best_metric:
            model_file = "_".join([opt.dataset, opt.model_name, 'e',str(opt.embedding_dim),'h', str(opt.lstm_hidden_dim)]) + '.pth'
            logger.info("save checkpoint: %s f1 score: %s", model_file, eval_f1)
            model_file = os.path.join(opt.ckpt_dir, model_file)
            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)
            torch.save({'state_dict': model.state_dict()},model_file)
            best_metric = eval_f1
            logger.info("Best checkpoint saved")
def eval():
    logger.info("start eval on test set")
    model.eval()
    eval_loss = []
    for batch, (sents, target_tags) in enumerate(test_loader):
        batch_size = sents.shape[0]
        sents = sents.to(device)
        target_tags = target_tags.to(device)
        logits = model(sents)
        predicted = logits.permute(0,2,1)
        loss = loss_fn(predicted, target_tags)
        eval_loss.append(loss.item())
        acc = acc_metric(predicted, target_tags)
        f1_score = f1_metric(predicted,target_tags)

    avg_loss =  np.average(eval_loss)
    acc = acc_metric.compute()
    f1 = f1_metric.compute()
    logger.info("*"*10 + ": eval result")
    logger.info(f"loss: {avg_loss}")
    logger.info(f"Accuracy:{acc}")
    logger.info(f"F1 Score: {f1}")
    acc_metric.reset()
    f1_metric.reset()
    return acc, f1
if __name__ == '__main__':

    opt = ParserInit().opt
    # figure.init_plt()
    acc_metric = torchmetrics.Accuracy(ignore_index=0)
    f1_metric = torchmetrics.F1Score(num_classes=opt.num_of_tags, multiclass=True, mdmc_average='samplewise',ignore_index=0)
    
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader, test_loader = get_loaders(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    train()
    