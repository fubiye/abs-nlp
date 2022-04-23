import logging
from absnlp.train.trainer import NerTrainer
logger = logging.getLogger(__name__)

class SoftmaxNerTrainer(NerTrainer):
    
    def __init__(self):
        super(NerTrainer, self).__init__()
    def train(self):
        logger.info("start train softmax NER model...")
        