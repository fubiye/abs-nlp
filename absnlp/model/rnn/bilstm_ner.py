import os
import torch 
import torch.nn as nn
from torch.nn import Module, Embedding, LSTM, Dropout, Linear, CrossEntropyLoss
from absnlp.model.losses.label_smoothing import LabelSmoothingCrossEntropy

class BiLstmSoftmaxModel(Module):
     
    def __init__(self, args):
        super(BiLstmSoftmaxModel, self).__init__()
        self.embedding = Embedding(args.vocab_size, args.embedding_dim,padding_idx=args.default_index)
        self.dropout = Dropout(args.dropout_rate)
        self.lstm = LSTM(args.embedding_dim, args.lstm_hidden_dim,bidirectional=args.lstm_bid)
        hidden_size = args.lstm_hidden_dim
        linear_input_size = (hidden_size * 2 if args.lstm_bid else hidden_size)
        self.linear = Linear(linear_input_size, args.num_labels)
        self.loss_type = args.loss_type
        self.ignore_index = args.default_index
        self.num_labels = args.num_labels
        
    
    def save_pretrained(self, output_dir):
        path = os.path.join(output_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), path)

    def init_weights(self, pretrained_embeddings):
        self.embedding.from_pretrained(pretrained_embeddings)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, input_ids, labels):
        embedding = self.embedding(input_ids)
        embedding = self.dropout(embedding)
        out_lstm, _ = self.lstm(embedding)
        logits =  self.linear(out_lstm)
        outputs = (logits,)

        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=self.ignore_index)
            # elif self.loss_type == 'focal':
            #     loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            # Only keep active parts of the loss
            
            loss = loss_fct(logits.view(-1, self.num_labels), labels.contiguous().view(-1))
            outputs = (loss, logits)
        return outputs  # (loss), scores, (hidden_states), (attentions)
        # return outputs
