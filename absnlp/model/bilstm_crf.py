import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torchmetrics.functional import f1_score
from torch import nn
from torchcrf import CRF

class BiLstmCrf(pl.LightningModule):

    def __init__(self, conf, vocab_sizes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({"conf": conf, "vocab_sizes": vocab_sizes})
        self.conf = conf
        self.encoder_conf = self.conf.model.encoder
        self.num_labels = vocab_sizes["ner_labels"]
        self.word_embeddings = nn.Embedding(
            vocab_sizes["words"], self.encoder_conf.input_size, padding_idx=0
        )
        self.encoder = instantiate(self.encoder_conf)
        hidden_size = self.encoder_conf.hidden_size
        bi_direction = self.encoder_conf.bidirectional
        linear_input_size = (hidden_size * 2 if bi_direction else hidden_size)
        self.dropout = nn.Dropout(self.encoder_conf.dropout)
        self.linear = nn.Linear(linear_input_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        # self.loss = nn.CrossEntropyLoss(ignore_index=0)
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward_train(self, sentences, labels):
        feats = self._get_lstm_features(sentences)
        mask = labels != 0
        loss = self.crf(feats, labels, mask=mask, reduction='mean')
        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result))
        foward_out = torch.stack(result_tensor)
        return -loss, foward_out.to(labels.device)

    def _get_lstm_features(self, sample):
        emb = self.word_embeddings(sample)
        emb = self.dropout(emb)
        out_lstm, _ = self.encoder(emb)
        out =  self.linear(out_lstm)
        return out
    
    def _decode(self, sentences):
        feats = self._get_lstm_features(sentences)
        results = self.crf.decode(feats)
        result_tensor = []
        for result in results:
            result_tensor.append(torch.tensor(result))
        return torch.stack(result_tensor)
    
    def forward(self, sample):
        return self._decode(sample)

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
        words = batch["words"]
        label = batch["labels"]
        loss, forward_output = self.forward_train(words, label)
        f1 = self._evaluate(forward_output, label)
        return loss, f1

    def _evaluate(self, forward_output, labels):
        # pred = self.softmax(logits)
        # pred = torch.argmax(logits, dim=-1)
        mask = labels != 0
        pred_no_pad, labels_no_pad = forward_output[mask], labels[mask]
        f1 = f1_score(pred_no_pad, labels_no_pad, num_classes=self.num_labels, average="macro")
        # loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return f1

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = instantiate(self.conf.train.optimizer, params=self.parameters())
        return optimizer
