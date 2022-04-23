import logging
import argparse
import os
logger = logging.getLogger(__name__)

class ParserInit():

    def __init__(self):
        logger.info("initializing training program...")
        self.parser = argparse.ArgumentParser()
        self.add_args()
        self.opt = self.parser.parse_args()
    
    def add_args(self):
        self.add_data_params()
        self.add_training_params()
        self.add_embedding_params()
        self.add_hyper_parameters()
        self.add_model_parameters()

    def add_data_params(self):
        home = os.path.expanduser('~')
        cache_path = os.path.join(home, '.cache')
        
        self.parser.add_argument('--cache_dir', default=cache_path, type=str, help="the path to store data")
        glove_cache_path = os.path.join(cache_path,'glove')
        self.parser.add_argument('--glove_cache',default=glove_cache_path,type=str, help="cache dir for glove")
        self.parser.add_argument('--dataset', default='conll2003', type=str, help="which dataset to load")
        self.parser.add_argument('--num_of_tags', default=9, help='total count of tags')
        
    def add_training_params(self):
        self.parser.add_argument('--epoches',default=10,type=int)
        self.parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    
    def add_embedding_params(self):
        self.parser.add_argument('--pretrain',default='glove',type=str, help="(glove|....)")
        self.parser.add_argument('--vector_name', default='6B', type=str)
        self.parser.add_argument('--embedding_dim',default=100, type=int, help="embedding size")
        
    def add_hyper_parameters(self):
        # Model Hyper Parameters
        # LSTM 
        self.parser.add_argument('--lstm_hidden_dim', default=100, type=int, help='hidden dim for lstm')

    def add_model_parameters(self):
        self.parser.add_argument('--ckpt_dir',default='checkpoints')
        self.parser.add_argument('--model_name', default='bert-softmax')




    
