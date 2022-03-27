from absnlp.util.args_util import ParserInit
from absnlp.util.data_loader import get_loaders
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

if __name__ == '__main__':
    opt = ParserInit().opt
    train_loader = get_loaders(opt)

