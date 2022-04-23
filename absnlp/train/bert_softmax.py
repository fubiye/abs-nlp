import logging
import os
import torch
from absnlp.model.transformers.model_ner import AutoModelForSoftmaxNer

from absnlp.train.trainer import NerTrainer
from absnlp.data.util import load_and_cache_examples

logger = logging.getLogger(__name__)


class SoftmaxNerTrainer(NerTrainer):
    
    def __init__(self, args):
        super(NerTrainer, self).__init__()
        self.args = args
        self.loss = torch.nn.CrossEntropyLoss()

    def main(self):
        logger.info("start train softmax NER model...")
        self.setup()
        if self.args.do_train:
            self.train()


    def init_model(self, args):
        self.model = AutoModelForSoftmaxNer.from_pretrained(
            args.pt_model_name_or_path,
            from_tf=bool(".ckpt" in args.pt_model_name_or_path),
            config=self.config,
            cache_dir=args.transformers_cache_dir,
        )
        self.model.to(args.device)
        
    def train(self):
        train_dataset = load_and_cache_examples(
                            self.args, 
                            self.tokenizer, 
                            self.labels, 
                            self.args.pad_token_label_id, 
                            mode='train')