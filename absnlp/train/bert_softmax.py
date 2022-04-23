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
        if self.args.do_train:
            self.train(self.args)


    def init_model(self, args):
        self.model = AutoModelForSoftmaxNer.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
            cache_dir=args.transformers_cache_dir,
        )
        self.model.to(args.device)
        
    def train(self, args):
        global_step, tr_loss = super().train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        self.tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        
        