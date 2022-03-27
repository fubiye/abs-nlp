import absnlp
from absnlp.util.args_util import ParserInit
import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y%m%d %H:%M:%S", level=logging.INFO)

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    logger.info("initializing training program...")
    parser = ParserInit().parser
    opt = parser.parse_args()
    logger.info("batch size: %d", opt.batch_size)
