import hydra
import pytorch_lightning as pl 

from omegaconf import DictConfig
from hydra.utils import instantiate 
from absnlp.data.data_module import DataModule
@hydra.main(config_path="config",config_name="application")
def train(config: DictConfig) -> None:
    pl.seed_everything(config.train.seed)
    data_module = DataModule(config)
    val_loader = data_module.val_dataloader()
    for batch, data in enumerate(val_loader):
        print(batch)
if __name__ == "__main__":
    train()