import os
import logging
import torch

from absnlp.train.trainer import GloveNerTrainer
from absnlp.model.rnn.bilstm_ner import BiLstmSoftmaxModel
from absnlp.model.rnn.model_ner import ModelLoader

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
            self.train(args)
        if args.do_eval:
            output_dir = os.path.join(args.output_dir, args.model_name)
            vocab = torch.load(os.path.join(output_dir,'pytorch_vocab.bin'))
            saved_args = torch.load(os.path.join(output_dir,'training_args.bin'))
            checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
            model_path = os.path.join(output_dir, 'pytorch_model.bin')
            model = ModelLoader.from_pretrained(saved_args, model_path)
            model.to(args.device)
            results, _ = self.eval(model, vocab, prefix='dev')
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                writer.write('***** Predict in dev dataset *****')
                writer.write("{} = {}\n".format('report', str(results['report'])))

    def init_model(self, args):
        self.model = BiLstmSoftmaxModel(args)
        self.model.init_weights(self.embeddings)
            
    def train(self, args):
        global_step, tr_loss = super().train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        output_dir = os.path.join(args.output_dir, args.model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model)  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(self.vocab, os.path.join(output_dir, 'pytorch_vocab.bin'))

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, "training_args.bin"))