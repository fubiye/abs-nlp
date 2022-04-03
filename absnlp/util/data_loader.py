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

        self.sents = [_ for doc in self.docs for _ in doc['sents'] if len(_) > 0] 
        self.sent_tags = [_ for doc in self.docs for _ in doc['sent_tags'] if len(_) > 0]

    def __getitem__(self, index):
        return self.sents[index], self.sent_tags[index]

    def __len__(self):
        return len(self.sents)

def collate_fn(batch):
    sents = [ _[0] for _ in batch]
    sent_tags = [ _[1] for _ in batch]
    return sents, sent_tags

def get_loader(opt, filepath):
    logger.info('loading data from: %s', filepath)
    dataset = ConllNERDataset(filepath)
    data_loader = data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,collate_fn=collate_fn)
    return data_loader

def get_loaders(opt):
    logger.info('loading data...')
    detect_data_path(opt)
    loader = get_loader(opt, opt.valid)
    for epoch in range(opt.epoches):
        logger.info("start epoch: %d", epoch)
        for batch, (sents, sent_tags) in enumerate(loader):
            logger.info('batch: %d, sents size: %d', batch, len(sents))
            for i in range(len(sents)):
                print('\t'.join(sents[i]))
                print('\t'.join(sent_tags[i]))
                print('\n')
                
            if batch > 0:
                return
        # for batch,(sents, sent_tags) in enumerate(loader):
        #     logger.info('batch: %d, sents size: %d', batch, len(sents))
    # sents, tags = dataset[:1]
    # for i in range(len(sents)):
    #     logger.info(" - %d \nsents: %s\ntokens: %s", i, ' '.join(sents[i]),' '.join(tags[i]))