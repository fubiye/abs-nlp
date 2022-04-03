import os
import logging
import torch
from torch.utils import data
from absnlp.dataset import conll
from absnlp.util.transform import collate_fn

logger = logging.getLogger(__name__)



def get_loader(opt, filepath):
    logger.info('loading data from: %s', filepath)
    dataset = conll.ConllNERDataset(filepath)
    data_loader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,collate_fn=lambda batch:collate_fn(batch,opt))
    return data_loader

def get_loaders(opt):
    logger.info('loading data...')
    conll.detect_data_path(opt)
    train_loader = get_loader(opt, opt.train)
    return train_loader
    # for epoch in range(opt.epoches):
    #     logger.info("start epoch: %d", epoch)
    #     for batch, (sents, sent_tags) in enumerate(loader):
    #         logger.info('batch: %d, sents size: %d', batch, len(sents))
    #         for i in range(len(sents)):
    #             print(sents[i])
    #             print(sent_tags[i])
    #             print('\n')
                
    #         if batch > 0:
    #             return
        # for batch,(sents, sent_tags) in enumerate(loader):
        #     logger.info('batch: %d, sents size: %d', batch, len(sents))
    # sents, tags = dataset[:1]
    # for i in range(len(sents)):
    #     logger.info(" - %d \nsents: %s\ntokens: %s", i, ' '.join(sents[i]),' '.join(tags[i]))