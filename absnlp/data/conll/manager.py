import os
from absnlp.data.data_module import BaseDataManager
from absnlp.data.conll.dataset import CoNLLDataset

class CoNLLDataManager(BaseDataManager):
    def __init__(self, 
        cache_dir, train_path, val_path, test_path, vocab_path,
        padding_size
        ):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        
        self.train_path = train_path
        self.train_file = os.path.join(self.cache_dir, self.train_path)
        self.val_path = val_path
        self.val_file = os.path.join(self.cache_dir, self.val_path)
        self.test_path = test_path
        self.test_file = os.path.join(self.cache_dir, self.test_path)
        self.padding_size = padding_size
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        if not os.path.exists(self.train_file):
            raise Exception(f"conll data not exists: {self.train_file}")
    
    def setup(self):
        self.train_dataset = CoNLLDataset(self.padding_size, self.train_file)
        vocabs = self.train_dataset.get_vocabs()
        self.val_dataset = CoNLLDataset(self.padding_size, self.val_file, vocabs)
        self.test_dataset = CoNLLDataset(self.padding_size, self.test_file, vocabs)
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset 

    def get_test_dataset(self):
        return self.test_dataset

    