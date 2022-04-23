import logging
import os
import torch
from absnlp.model.transformers.model_ner import AutoModelForSoftmaxNer

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

    def init_model(self, args):
        self.model = AutoModelForSoftmaxNer.from_pretrained(
            args.pt_model_name_or_path,
            from_tf=bool(".ckpt" in args.pt_model_name_or_path),
            config=self.config,
            cache_dir=args.transformers_cache_dir,
        )
        self.model.to(args.device)
        
    def train(self):
        pass