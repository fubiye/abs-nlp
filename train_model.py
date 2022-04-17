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
from tqdm import tqdm

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
        t = tqdm(train_loader)
        for batch, (sents, target_tags) in enumerate(t):
            batch_size = sents.shape[0]
            sents = sents.to(device)
            target_tags = target_tags.to(device)
            logits = model(sents)
            predicted = logits.permute(0,2,1)
            loss = loss_fn(predicted, target_tags)
            total_loss += loss.item()
            train_loss.append(loss.item())
            
            # print(f'target: {target_tags.device}') print(f'predicted: {predicted.device}')
            predicted_tags = torch.argmax(predicted,dim=1)
            acc = acc_metric(predicted_tags, target_tags)
            f1_score = f1_metric(predicted_tags,target_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (batch + 1) % 10 == 0:
                cur_loss = total_loss
                acc = acc_metric.compute()
                f1 = f1_metric.compute()
                loss_history.append(cur_loss)
                f1_history.append(f1)
                total_loss = 0
                num_ele = batch * batch_size
                t.set_description(f"epochs : {epoch}, batch : {batch}, loss : {cur_loss:.4f}, f1 : {f1:.4f}, acc: {acc:.4f}")
                t.refresh()
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
        eval_acc, eval_f1 = eval()
        if eval_acc > best_metric:
            model_file = "_".join([opt.dataset, opt.model_name, 'e',str(opt.embedding_dim),'h', str(opt.lstm_hidden_dim)]) + '.pth'
            logger.info("save checkpoint: %s f1 score: %s", model_file, eval_f1)
            model_file = os.path.join(opt.ckpt_dir, model_file)
            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)
            torch.save({'state_dict': model.state_dict()},model_file)
            best_metric = eval_acc
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
        predicted_tags = torch.argmax(predicted,dim=1)
        acc = acc_metric(predicted_tags, target_tags)
        f1_score = f1_metric(predicted_tags,target_tags)

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = ParserInit().opt
    # figure.init_plt()
    acc_metric = torchmetrics.Accuracy(ignore_index=0).to(device)
    f1_metric = torchmetrics.F1Score(num_classes=opt.num_of_tags,threshold=0, multiclass=True,average='macro', mdmc_average='global',ignore_index=0).to(device)
    
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader, test_loader = get_loaders(opt)
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    train()
    
