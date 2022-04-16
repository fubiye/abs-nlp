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
    test_loader = get_loader(opt, opt.valid)
    return train_loader, test_loader