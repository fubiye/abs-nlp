# https://github.com/pytorch/text/issues/1350
from torchtext.vocab import GloVe, vocab
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer

import torch
import torch.nn as nn

class TextClassificationModel(nn.Module):

    def __init__(self, pretrained_embeddings, num_class, freeze_embeddings=False):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(pretrained_embeddings,sparse=True)
        self.fc = nn.Linear(pretrained_embeddings.shape[1], num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self,text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

train_iter = AG_NEWS(split='train')
num_class = len(set([label for (label,_) in train_iter]))
unk_token = "<unk>"
unk_index = 0
glove_vectors = GloVe(name='6B', dim='50', cache='d:\dataset\glove')
glove_vocab = vocab(glove_vectors.stoi)
glove_vocab.insert_token(unk_token, unk_index)
glove_vocab.set_default_index(unk_index)
pretrained_embeddings = glove_vectors.vectors 
pretrained_embeddings = torch.cat((torch.zeros(1, pretrained_embeddings.shape[1]),pretrained_embeddings))

glove_model = TextClassificationModel(pretrained_embeddings, num_class)

tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split='train')
example_text = next(iter(train_iter))[1]
print(example_text)
tokens = tokenizer(example_text)
indices = glove_vocab(tokens)
text_input = torch.tensor(indices)
offset_input = torch.tensor([0])

print(text_input)
model_output = glove_model(text_input, offset_input)
print(model_output)