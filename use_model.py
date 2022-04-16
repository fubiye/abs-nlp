from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
from absnlp.vocab.pretrain_static import PAD_IDX, get_vocab
from absnlp.dataset.conll import preprocess_tags
from absnlp.model.rnn import SimpleRnnNet
import absnlp.util.figure as figure
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
import numpy as np
import os 
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    opt = ParserInit().opt   
    opt.vocab, opt.vectors = get_vocab(opt)
    opt.tag2id, opt.id2tag = preprocess_tags()
    train_loader, test_loader = get_loaders(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.vocab_size = len(opt.vocab)
    model = SimpleRnnNet(opt)
    model_file = "_".join([opt.dataset, opt.model_name, 'e',str(opt.embedding_dim),'h', str(opt.lstm_hidden_dim)]) + '.pth'
    model_file = os.path.join(opt.ckpt_dir, model_file)
    logger.info("load model from file: %s", model_file)
    model.load_state_dict(torch.load(model_file, map_location='cpu')['state_dict'])
    sents = [
        "West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship .",
        "Germany 's representative to the European Union 's veterinary committee Werner Zwingmann said on Wednesday consumers should buy sheepmeat from countries other than Britain until the scientific advice was clearer ."
        ]
    ids = [opt.vocab(sent.split()) for sent in sents]
    max_len = np.max([len(tokens) for tokens in ids])
    padded_ids = []
    for tokens in ids:
        if not len(tokens) < max_len:
            padded_ids.append(tokens[:max_len])
            continue
        padded_ids.append(tokens + [PAD_IDX] * (max_len - len(tokens)))
    
    logits = model(torch.tensor(padded_ids))
    output = torch.argmax(logits, dim=2)
    print(sents)
    for sent in output:
        print([opt.id2tag[tagid.item()] for tagid in sent])
