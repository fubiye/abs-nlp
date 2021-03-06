import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from absnlp.util.ner import get_labels, collate_fn
from absnlp.data.util import load_and_cache_examples, load_dataset_with_vocab
from absnlp.util.vocab import load_glove
from absnlp.util.metrics import get_entities_bio, f1_score, classification_report

logger = logging.getLogger(__name__)

TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

class NerTrainer():
    def __init__(self, args):
        pass
    
    def train(self):
        args = self.args
        self.tb_writer = SummaryWriter(args.output_dir)
        self.train_dataset = self.load_dataset('train')
        self.train_sampler = RandomSampler(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset,
                                  sampler=self.train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=collate_fn)
        self.t_total = self.calc_training_steps(args)
        self.optimizer, self.scheduler = self.prepare_optimizer_and_scheduler(args)
        self.print_training_info(args)
        self.init_training_metrics()
        self.model.zero_grad()
        train_iterator = trange(
            self.epochs_trained, int(args.num_train_epochs), desc="Epoch"
        )
        for _ in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if self.steps_trained_in_current_epoch > 0:
                    self.steps_trained_in_current_epoch -= 1
                    continue
                self.model.train()
                batch = tuple(t.to(args.device) for t in batch)
                loss = self.batch_forward(args, batch)
                loss.backward()
                self.tr_loss += loss.item()
                epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    self.global_step += 1

                    if args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
                        # Log metrics
                        if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                            results, _ = self.eval(self.model, self.vocab, self.global_step)
                            for key, value in results.items():
                                if isinstance(value, float) or isinstance(value, int):
                                    self.tb_writer.add_scalar("eval_{}".format(key), value, self.global_step)
                        self.tb_writer.add_scalar("lr", self.scheduler.get_lr()[0], self.global_step)
                        self.tb_writer.add_scalar("loss", (self.tr_loss - self.logging_loss) / args.logging_steps, self.global_step)
                        self.logging_loss = self.tr_loss

                        if self.best_score < results['f1']:
                            self.best_score = results['f1']
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            self.save_model(output_dir)

            if args.max_steps > 0 and self.global_step > args.max_steps:
                train_iterator.close()
        
        self.tb_writer.close()
        return self.global_step, self.tr_loss / self.global_step
    def load_dataset(self, mode):
        pass
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
        pass
    
    def print_training_info(self, args):
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.batch_size
            * args.gradient_accumulation_steps,
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

    def init_training_metrics(self):
        self.global_step = 0
        self.epochs_trained = 0
        self.best_score = 0.0
        self.steps_trained_in_current_epoch = 0
        self.try_continue_training(self.args)
        self.tr_loss, self.logging_loss = 0.0, 0.0
    
    def try_continue_training(self,args):
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            try:
                self.global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                self.global_step = 0
            self.epochs_trained = self.global_step // (len(self.train_dataloader) // args.gradient_accumulation_steps)
            self.steps_trained_in_current_epoch = self.global_step % (len(self.train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", self.epochs_trained)
            logger.info("  Continuing training from global step %d", self.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", self.steps_trained_in_current_epoch)

    def evaluate(self, args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.batch_size)
        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        trues = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "valid_mask": batch[2],
                        "labels": batch[4], }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don"t use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                trues = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                trues = np.append(trues, inputs["labels"].detach().cpu().numpy(), axis=0)
            
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)
        label_map = {i: label for i, label in enumerate(labels)}
        
        trues_list = [[] for _ in range(trues.shape[0])]
        preds_list = [[] for _ in range(preds.shape[0])]

        for i in range(trues.shape[0]):
            for j in range(trues.shape[1]):
                if trues[i, j] != pad_token_label_id:
                    trues_list[i].append(label_map[trues[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        true_entities = get_entities_bio(trues_list)
        pred_entities = get_entities_bio(preds_list)
        results = {
            "loss": eval_loss,
            "f1": f1_score(true_entities, pred_entities),
            'report': classification_report(true_entities, pred_entities)
        }
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            writer.write("***** Eval results {} *****\n".format(prefix))
            writer.write("***** Eval loss : {} *****\n".format(eval_loss))
            for key in sorted(results.keys()):
                if key == 'report_dict':
                    continue
                logger.info("{} = {}".format(key, str(results[key])))
                writer.write("{} = {}\n".format(key, str(results[key])))
        return results, preds_list

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
        self.args.default_index = self.loss.ignore_index
        self.args.pad_token_label_id = self.loss.ignore_index
        self.model_config(self.args)
        self.init_tokenizer(self.args)

    def model_config(self, args):
        pass

    def init_tokenizer(self,args):
        pass
    
    def init_train_tokenizer(self, args):
        pass

    def init_model(self, args):
        pass
    
    def batch_forward(self,args, batch):
        pass
    def eval(self, args):
        pass
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.do_save_model(output_dir)

    def do_save_model(self, output_dir):
        pass

class TransformerNerTrainer(NerTrainer):
    def __init__(self, args):
        super(NerTrainer, self).__init__(args)
    
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
        self.tokenizer_args = {k: v for k, v in vars(self.args).items() if v is not None and k in TOKENIZER_ARGS}
        logger.info("Tokenizer arguments: %s", self.tokenizer_args)
    
    def init_train_tokenizer(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.transformers_cache_dir,
            **self.tokenizer_args,
        )
    
    def load_dataset(self, mode):
        return load_and_cache_examples(
                            self.args, 
                            self.tokenizer, 
                            self.labels, 
                            self.args.pad_token_label_id, 
                            mode)
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

    def batch_forward(self, args, batch):
        inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "valid_mask": batch[2],
                      "labels": batch[4], }
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[3] if args.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use segment_ids
        outputs = self.model(**inputs)
        loss = outputs[0] 
        return loss

    def eval(self, model, vocab, prefix):
        return self.evaluate(self.args, self.model, self.tokenizer, self.labels, args.pad_token_label_id, mode="dev",
                                                prefix=prefix)
    
    def do_save_model(self, output_dir):
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

class GloveNerTrainer(NerTrainer):
    def __init__(self, args):
        super(GloveNerTrainer, self).__init__(args)
    
    def init_train_tokenizer(self, args):
        self.vocab, self.embeddings = load_glove(args)
        args.vocab_size = len(self.vocab)
    
    def load_dataset(self, mode):
        return load_dataset_with_vocab(
            self.args, 
            self.vocab, 
            self.labels, 
            self.args.pad_token_label_id, 
            mode)

    def prepare_optimizer_and_scheduler(self, args):
        
        parameters = self.model.parameters()
        
        optimizer = SGD(parameters, lr=args.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        # # Check if saved optimizer or scheduler states exist
        # output_dir = os.path.join(args.output_dir, args.model_name)
        # if os.path.isfile(os.path.join(output_dir, "optimizer.pt")) and os.path.isfile(os.path.join(output_dir, "scheduler.pt")):
        #     # Load in optimizer and scheduler states
        #     optimizer.load_state_dict(torch.load(os.path.join(output_dir, "optimizer.pt")))
        #     scheduler.load_state_dict(torch.load(os.path.join(output_dir, "scheduler.pt")))
        return optimizer, scheduler
    
    def batch_forward(self, args, batch):
        inputs = {
            "input_ids": batch[0],
            "labels": batch[1]
        }
        outputs = self.model(**inputs)
        loss = outputs[0] 
        return loss
    def eval(self, model, vocab, prefix):
        args, labels, pad_token_label_id = self.args, self.labels, self.args.pad_token_label_id,
        eval_dataset = load_dataset_with_vocab(
            args, 
            vocab, 
            labels, 
            pad_token_label_id,
            'dev'
            )
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.batch_size)
        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        trues = None
        model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "labels": batch[1]
                }
                
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]                
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                trues = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                trues = np.append(trues, inputs["labels"].detach().cpu().numpy(), axis=0)
            
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)
        label_map = {i: label for i, label in enumerate(labels)}
        
        trues_list = [[] for _ in range(trues.shape[0])]
        preds_list = [[] for _ in range(preds.shape[0])]

        for i in range(trues.shape[0]):
            for j in range(trues.shape[1]):
                if trues[i, j] != pad_token_label_id:
                    trues_list[i].append(label_map[trues[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])
        true_entities = get_entities_bio(trues_list)
        pred_entities = get_entities_bio(preds_list)
        results = {
            "loss": eval_loss,
            "f1": f1_score(true_entities, pred_entities),
            'report': classification_report(true_entities, pred_entities)
        }
        output_dir = os.path.join(args.output_dir, args.model_name)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            writer.write("***** Eval results {} *****\n".format(prefix))
            writer.write("***** Eval loss : {} *****\n".format(eval_loss))
            for key in sorted(results.keys()):
                if key == 'report_dict':
                    continue
                logger.info("{} = {}".format(key, str(results[key])))
                writer.write("{} = {}\n".format(key, str(results[key])))
        return results, preds_list
    
    def do_save_model(self, output_dir):

        torch.save(self.model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.vocab, os.path.join(output_dir, 'pytorch_model.bin'))

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)