import os
import logging
import torch
import torch.utils.data as data 
logger = logging.getLogger(__name__)

def detect_data_path(opt):
    opt.train = os.path.join(opt.cache_dir, opt.dataset, 'train.txt')
    opt.test = os.path.join(opt.cache_dir, opt.dataset, 'test.txt')
    opt.valid = os.path.join(opt.cache_dir, opt.dataset, 'valid.txt')

    if not (os.path.exists(opt.train)):
        logger.error('data not found in path: {}', opt.train)
        raise Exception("data not found")

class ConllNERDataset(data.Dataset):
    
    def __init__(self, filepath):
        self.filepath = filepath
        lines = self.load_data_from_file()
        self.convert_to_docs(lines)

    def load_data_from_file(self):
        with open(self.filepath,'r',encoding='utf-8') as f:
            lines = f.readlines()
            return [line.strip() for line in lines]

    def convert_to_docs(self, lines):
        self.docs = []
        doc, sent, tags = None, None, None

        for line in lines:
            # line seprator
            if len(line) == 0:
                if sent is not None:
                    doc['sents'].append(sent)
                sent = []

                if tags is not None:
                    doc['sent_tags'].append(tags)
                tags = []

                continue

            tokens = line.split()
            
            # document seprator
            if '-DOCSTART-' == tokens[0]:
                if doc is not None:
                    self.docs.append(doc)
                doc = {
                    'sents': [],
                    'sent_tags': []
                }
                continue
            
            sent.append(tokens[0])
            tags.append(tokens[3])

        self.sents = [_ for doc in self.docs for _ in doc['sents']]
        self.sent_tags = [_ for doc in self.docs for _ in doc['sent_tags']]

    def __getitem__(self, index):
        return (self.sents[index], self.sent_tags[index])

    def __len__(self):
        return len(self.sents)

def get_loader(filepath):
    logger.info('loading data from: %s', filepath)
    dataset = ConllNERDataset(filepath)
    return dataset

def get_loaders(opt):
    logger.info('loading data...')
    detect_data_path(opt)
    dataset = get_loader(opt.valid)
    sents, tags = dataset[:10]
    for i in range(len(sents)):
        logger.info(" - %d \nsents: %s\ntokens: %s", i, ' '.join(sents[i]),' '.join(tags[i]))