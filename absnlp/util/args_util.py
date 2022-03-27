import argparse

class ParserInit():

    def __init__(self):
        self.parser = parser = argparse.ArgumentParser()
        self.add_args()
    
    def add_args(self):
        self.add_hyper_parameters()

    def add_hyper_parameters(self):
        self.parser.add_argument('--batch_size', default=4, type=int, help='batch size')


    