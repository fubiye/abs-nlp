from tkinter import S
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRnnNet(nn.Module):

    def __init__(self, opt) -> None:
        super(SimpleRnnNet, self).__init__()
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.lstm = nn.LSTM(opt.embedding_dim, opt.lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(opt.lstm_hidden_dim, opt.num_of_tags)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        out, _ = self.lstm(embedded)
        out = self.fc(out)
        out = torch.argmax(out, dim=2)
        return out
