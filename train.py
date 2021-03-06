import hydra
import torch
import pytorch_lightning as pl 

from omegaconf import DictConfig
from hydra.utils import instantiate 
from absnlp.data.data_module import DataModule
from absnlp.model.base import BaseNerModule
from absnlp.model.bilstm_crf import BiLstmCrf
from absnlp.model.bert import BertNerModule
from absnlp.train.manager import TrainManager
from absnlp.train.manager import TRAINERS
from absnlp.util.init import  bootstrap
# @hydra.main(config_path="config",config_name="application")
# def train(config: DictConfig) -> None:
#     pl.seed_everything(config.train.seed)
#     data_module = DataModule(config)
#     vocab_sizes = data_module.get_vocab_sizes()
#     ner = BertNerModule(config, vocab_sizes)
#     train_manager = TrainManager(config)
#     gpus = config.train.trainer.gpus if torch.cuda.is_available() else 0
#     trainer: Trainer = instantiate(
#         config.train.trainer, callbacks=train_manager.callbacks, gpus=gpus, fast_dev_run=False,
#     )
#     trainer.fit(ner, datamodule=data_module)
#     trainer.test(ner, datamodule=data_module)

if __name__ == "__main__":
    args = bootstrap()
    trainer = TRAINERS[args.model_name](args)
    trainer.main()

