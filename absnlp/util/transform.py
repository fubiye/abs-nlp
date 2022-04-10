from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

from absnlp.vocab.pretrain_static import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func 

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def text_transform(opt):
    vocab_transform = lambda sents: opt.vocab(sents)    
    return sequential_transforms(vocab_transform, tensor_transform)

def label_transform(opt):
    tag_id_transform = lambda tags: [opt.tag2id[tag] for tag in tags]
    tag_tensor_transform = lambda tokens: torch.cat((torch.tensor([0]),torch.tensor(tokens), torch.tensor([0])))
    return sequential_transforms(tag_id_transform, tag_tensor_transform)

def collate_fn(batch, opt):
    sent_transform = text_transform(opt)
    tag_transform = label_transform(opt)

    sents, tags = [], []
    for sentences, sent_tags in batch:
        sents.append(sent_transform(sentences))
        tags.append(tag_transform(sent_tags))
    sents = pad_sequence(sents, padding_value=PAD_IDX,batch_first=True)
    tags = pad_sequence(tags, batch_first=True)
    
    return sents, tags