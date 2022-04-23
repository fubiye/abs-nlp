import logging
import os
import torch
from absnlp.train.trainer import NerTrainer

logger = logging.getLogger(__name__)


class SoftmaxNerTrainer(NerTrainer):
    
    def __init__(self, args):
        super(NerTrainer, self).__init__()
        self.args = args
        self.loss = torch.nn.CrossEntropyLoss()

    def main(self):
        logger.info("start train softmax NER model...")
        self.setup()

    
    def train(self):
        pass