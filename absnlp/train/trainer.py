import os
import logging
import torch
from torch.utils.data import RandomSampler, DataLoader
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
from absnlp.util.ner import get_labels, collate_fn
from absnlp.data.util import load_and_cache_examples

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

class NerTrainer():
    def __init__(self, args):
        pass
    def train(self):
        args = self.args
        self.tb_writer = SummaryWriter(args.output_dir)
        self.train_dataset = load_and_cache_examples(
                            args, 
                            self.tokenizer, 
                            self.labels, 
                            args.pad_token_label_id, 
                            mode='train')
        self.train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset,
                                  sampler=self.train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn)
        self.t_total = self.calc_training_steps(args)
        optimizer, scheduler = self.prepare_optimizer_and_scheduler(args)

    def calc_training_steps(self, args):
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(self.train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.logging_steps = eval(args.logging_steps)
        if isinstance(args.logging_steps, float):
            args.logging_steps = int(args.logging_steps * len(self.train_dataloader)) // args.gradient_accumulation_steps
        return t_total
    def prepare_optimizer_and_scheduler(self, args):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = eval('self.model.{}'.format(args.model_type)).named_parameters()
        classifier_parameters = self.model.classifier.named_parameters()
        args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
        args.classifier_lr = args.classifier_lr if args.classifier_lr else args.learning_rate
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.bert_lr},

            {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.classifier_lr},
            {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.classifier_lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=self.t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
        return optimizer, scheduler
    def eval(self):
        pass

    def predict(self):
        pass
    
    def setup(self):
        self.prepare_dataset()
        self.prepare_training()
        logger.info("Training/evaluation parameters %s", self.args)

    def prepare_dataset(self):
        labels_path = os.path.join(self.args.cache_dir, self.args.dataset, self.args.labels)
        self.labels = get_labels(labels_path)
        self.num_labels = len(self.labels)
        logger.info("dataset: %s labels(%d): %s", self.args.dataset, self.num_labels, ', '.join(self.labels))
    
    def prepare_training(self):
        self.args.pad_token_label_id = self.loss.ignore_index
        self.model_config(self.args)
        self.init_tokenizer(self.args)
        self.init_model(self.args)

    def model_config(self, args):
        self.config = AutoConfig.from_pretrained(
            args.model_name_or_path,
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
            args.model_name_or_path,
            cache_dir=args.transformers_cache_dir,
            **tokenizer_args,
        )
    def init_model(self, args):
        pass