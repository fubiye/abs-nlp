import sys
sys.path.append("../")
import absnlp
from absnlp.models.ner.BiLSTMCrf import BiLSTM_CRF

if __name__ == '__main__':
    BiLSTM_CRF.train()