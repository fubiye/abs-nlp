import pytorch_lightning as pl 
from hydra.utils import instantiate 

class BaseDataManager():
    def __init__(self):
        pass
    def prepare_data(self):
        pass

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.conf = config
        self.manager = instantiate(config.data.manager)