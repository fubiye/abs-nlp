import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torchmetrics.functional import f1_score
from torch import nn
from transformers import BertModel

class BertNerModule(pl.LightningModule):

    def __init__(self, conf, vocab_sizes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({"conf": conf, "vocab_sizes": vocab_sizes})
        self.conf = conf
        self.encoder_conf = self.conf.model.encoder
        self.num_labels = vocab_sizes["ner_labels"]
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(self.encoder_conf.dropout)
        self.linear = nn.Linear(768, self.num_labels)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sample):
        outputs = self.bert(**sample)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return self.linear(sequence_output)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, f1 = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        loss, f1 = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_f1", f1)
        return loss

    def test_step(self, batch: dict, batch_idx: int):
        _, f1 = self._shared_step(batch, batch_idx)
        self.log("test_f1", f1)

    def _shared_step(self, batch: dict, batch_idx: int):
        words = batch["input_ids"]
        label = batch["labels"]
        forward_output = self.forward(words)
        loss, f1 = self._evaluate(forward_output, label)
        return loss, f1

    def _evaluate(self, logits, labels):
        pred = self.softmax(logits)
        pred = torch.argmax(pred, dim=-1)
        mask = labels != 0
        pred_no_pad, labels_no_pad = pred[mask], labels[mask]
        f1 = f1_score(pred_no_pad, labels_no_pad, num_classes=self.num_labels, average="macro")
        loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss, f1

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = instantiate(self.conf.train.optimizer, params=self.parameters())
        return optimizer