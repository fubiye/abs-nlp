import logging
import torch

from absnlp.train.trainer import GloveNerTrainer
from absnlp.model.rnn.bilstm_ner import BiLstmSoftmaxModel

logger = logging.getLogger(__name__)

class BiLstmSoftmaxNerTrainer(GloveNerTrainer):
    
    def __init__(self, args):
        super(BiLstmSoftmaxNerTrainer, self).__init__(args)
        self.args = args
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def main(self):
        logger.info("start train Bi-LSTM softmax NER model...")
        self.setup()
        args = self.args
        if args.do_train:
            self.init_train_tokenizer(args)
            self.init_model(args)
            # self.train(args)

    def init_model(self, args):
        self.model = BiLstmSoftmaxModel(args)