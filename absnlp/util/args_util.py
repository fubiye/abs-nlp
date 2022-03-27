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
        self.add_training_parameters()
        self.add_hyper_parameters()

    def add_training_parameters(self):
        self.parser.add_argument('--cache_dir', default='d:\dataset', type=str, help="the path to store data")
        self.parser.add_argument('--dataset', default='conll2003', type=str, help="which dataset to load")

    def add_hyper_parameters(self):
        self.parser.add_argument('--batch_size', default=4, type=int, help='batch size')


    