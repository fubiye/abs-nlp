import os
from absnlp.data.data_module import BaseDataManager

class CoNLLDataManager(BaseDataManager):
    def __init__(self, cache_dir, train_path, val_path, test_path):
        super().__init__()
        self.cache_dir = cache_dir
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        
        self.prepare_data()
        self.setup()
    
    def prepare_data(self):
        data_dir = os.path.expanduser(self.cache_dir)
        if not os.path.exists(os.path.join(data_dir, self.train_path)):
            raise Exception("data not exists")
    
    def setup(self):
        pass