from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from hydra.utils import instantiate
from absnlp.train.bert_softmax import SoftmaxNerTrainer
from absnlp.train.bilstm_softmax import BiLstmSoftMaxNerTrainer

class TrainManager:

    def __init__(self, conf: DictConfig):
        self.conf = conf
        self.callbacks = []
        self.add_callbacks()

    def add_callbacks(self):
        train_conf = self.conf.train
        if train_conf.early_stopping is not None:
            early_stopping: EarlyStopping = instantiate(train_conf.early_stopping)
            self.callbacks.append(early_stopping)

        if train_conf.model_checkpoint is not None:
            model_checkpoint: ModelCheckpoint = instantiate(train_conf.model_checkpoint)
            self.callbacks.append(model_checkpoint)

TRAINERS = {
    'bilstm-softmax': BiLstmSoftMaxNerTrainer,
    'bert-softmax': SoftmaxNerTrainer
}