import pytorch_lightning as pl 
from hydra.utils import instantiate 
from torch.utils.data import DataLoader

class BaseDataManager():
    def __init__(self):
        pass
    def prepare_data(self):
        pass
    def setup(self):
        pass
    def get_train_dataset(self):
        pass

    def get_val_dataset(self):
        pass

    def get_test_dataset(self):
        pass
class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.manager: BaseDataManager  = instantiate(config.data.manager)
    def prepare_data(self):
        self.manager.prepare_data()
    def setup(self, stage):
        self.manager.setup()
    def train_dataloader(self):
        return DataLoader(
            self.manager,get_train_dataset(),
            num_workers=self.config.data.num_workers,
            batch_size = self.config.data.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.manager.get_val_dataset(),
            num_workers=self.config.data.num_workers,
            batch_size = self.config.data.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        return DataLoader(
            self.manager.get_test_dataset(),
            num_workers=self.config.data.num_workers,
            batch_size = self.config.data.batch_size,
            shuffle=False
        )