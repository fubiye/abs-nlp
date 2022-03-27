import logging
import argparse
logger = logging.getLogger(__name__)

class ParserInit():

    def __init__(self):
        logger.info("initializing training program...")
        self.parser = parser = argparse.ArgumentParser()
        self.add_args()
        self.opt = self.parser.parse_args()
    
    def add_args(self):
        self.add_data_params()
        self.add_training_params()
        # self.add_hyper_parameters()

    def add_data_params(self):
        self.parser.add_argument('--cache_dir', default='d:\dataset', type=str, help="the path to store data")
        self.parser.add_argument('--dataset', default='conll2003', type=str, help="which dataset to load")
        
    def add_training_params(self):
        self.parser.add_argument('--epoches',default=1,type=int)
        self.parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    # def add_hyper_parameters(self):
    #     self.parser.add_argument('--batch_size', default=4, type=int, help='batch size')


    