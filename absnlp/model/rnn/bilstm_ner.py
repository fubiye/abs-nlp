import torch 
from torch.nn import Module, Embedding, LSTM, Dropout, Linear

class BiLstmSoftmaxModel(Module):
    
    def __init__(self, args):
        super(BiLstmSoftmaxModel, self).__init__()
        self.embedding = Embedding(args.vocab_size, args.embedding_dim,padding_idx=args.default_index)
        self.dropout = Dropout(args.dropout_rate)
        self.lstm = LSTM(args.embedding_dim, args.lstm_hidden_dim,bidirectional=args.lstm_bid)
        hidden_size = args.lstm_hidden_dim
        linear_input_size = (hidden_size * 2 if args.lstm_bid else hidden_size)
        self.fc = Linear(linear_input_size, args.num_labels)