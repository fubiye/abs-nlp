import logging
import torch
from absnlp.util.args_util import ParserInit

logger = logging.getLogger(__name__)

def bootstrap():
    config_logging()
    logger.info("start bootstraping application")
    args = parse_arguments()
    detect_device(args)
    return args

def parse_arguments():
    args = ParserInit().opt
    return args
def config_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

def detect_device(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    logger.info("using device: %s", device)
