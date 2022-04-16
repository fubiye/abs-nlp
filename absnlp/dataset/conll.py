import os 
import logging
from torch.utils import data 

logger = logging.getLogger(__name__)

def detect_data_path(opt):
    opt.train = os.path.join(opt.cache_dir, opt.dataset, 'train.txt')
    opt.test = os.path.join(opt.cache_dir, opt.dataset, 'test.txt')
    opt.valid = os.path.join(opt.cache_dir, opt.dataset, 'valid.txt')

    if not (os.path.exists(opt.train)):
        logger.error('data not found in path: {}', opt.train)
        raise Exception("data not found")
def preprocess_tags():
    raw_tags = ['PER','LOC','ORG','MISC']
    _tags = ['<pad>','O']
    for tag in raw_tags:
        _tags += ['B-'+tag, 'I-'+tag]
    tags = _tags
    tag2id = {tag: index for index, tag in enumerate(tags)}
    id2tag = {index: tag for index, tag in enumerate(tags)}
    return tag2id, id2tag
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