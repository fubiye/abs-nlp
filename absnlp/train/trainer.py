import os
import logging
import torch
from transformers import AutoConfig, AutoTokenizer
logger = logging.getLogger(__name__)
from absnlp.util.ner import get_labels

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]
class NerTrainer():
    def __init__(self, args):
        pass
    def train(self):
        pass
    
    def eval(self):
        pass

    def predict(self):
        pass
    
    def setup(self):
        self.prepare_dataset()
        self.prepare_training()

    def prepare_dataset(self):
        labels_path = os.path.join(self.args.cache_dir, self.args.dataset, self.args.labels)
        self.labels = get_labels(labels_path)
        self.num_labels = len(self.labels)
        logger.info("dataset: %s labels(%d): %s", self.args.dataset, self.num_labels, ', '.join(self.labels))
    
    def prepare_training(self):
        self.args.pad_token_label_id = self.loss.ignore_index
        self.model_config(self.args)
        self.init_tokenizer(self.args)

    def model_config(self, args):
        self.config = AutoConfig.from_pretrained(
            args.pt_model_name_or_path,
            num_labels=self.num_labels,
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
            cache_dir=args.transformers_cache_dir,
        )
        #####
        setattr(self.config, 'loss_type', args.loss_type)
        #####
    def init_tokenizer(self,args):
        tokenizer_args = {k: v for k, v in vars(self.args).items() if v is not None and k in TOKENIZER_ARGS}
        logger.info("Tokenizer arguments: %s", tokenizer_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pt_model_name_or_path,
            cache_dir=args.transformers_cache_dir,
            **tokenizer_args,
        )