import logging

from absnlp.util.args_util import ParserInit

def bootstrap():
    args = parse_arguments()
    config_logging()
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
