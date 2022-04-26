import os
from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchmetrics.functional import f1_score
# tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[2])
# word_ids = tokenized_inputs.word_ids(batch_index=2)
# for (token, word_id) in zip(tokens, word_ids):
#     print(f'{token}\t{word_id}')
from torchtext.vocab import GloVe, vocab

glove_cache = os.path.expanduser("~/.cache/glove")
glove = GloVe(name='6B', dim=100, cache=glove_cache)
glove_vocab = vocab(glove.stoi, specials=['<unk>'])
glove_vocab.set_default_index(0)

vectors = glove.vectors
vectors = torch.cat((torch.zeros(1, vectors.shape[1]),vectors))

class ExampleDataset(Dataset):
    sentences = []
    tags = []
    samples = []
    def __init__(self):
        self.sentences = [
            ["OMERS","is","one","of","Canada","'","s","largest","pension","funds","with","over","$","85",".","2","billion","of","net","assets","as","of","year","-","end","2016","."],
            ["Oxford","Properties","Group","is","the","global","real","estate","investment","arm","of","OMERS","."],
            ["Established","in","1960",",","Oxford","manages","real","estate","for","itself","and","on","behalf","of","its","co","-","owners","and","investment","partners","with","offices","across","Canada","and","in","New","York",",","Washington",",","Boston",",","London","and","Luxembourg","."],
            ["Oxford","'","s","approximately","$","41",".","0","billion","real","estate","portfolio","consists","of","approximately","60","million","sq",".","ft",".","and","over","150","properties","that","total","approximately","3",",","600","hotels","rooms","and","over","9",",","500","residential","units","located","across","Canada",",","Western","Europe","and","US","markets","."],
            ["The","Olympic","Tower","Whole","Loan","has","a","10","-","year","term","and","pays","interest","only","for","the","term","of","the","loan","."],
            ["The","Olympic","Tower","Whole","Loan","accrues","interest","at","a","fixed","rate","equal","to","3",".","95394737","%","per","annum","and","has","a","Cut","-","off","Date","Balance","of","$","80",".","0","million","."],
            ["The","Olympic","Tower","Whole","Loan","proceeds",",","in","addition","to","a","$","240",".","0","million","mezzanine","loan",",","were","used","to","refinance","existing","debt","of","approximately","$","249",".","9","million",",","fund","approximately","$","61",".","6","million","in","upfront","reserves",",","pay","transaction","costs","of","approximately","$","22",".","7","million","and","return","approximately","$","665",".","8","million","of","equity","to","the","owners","of","the","borrower","."]
        ]
        self.tags = [
            ['B-ORG','O','O','O','B-LOC','O','O','O','O','O','O','O','B-MONEY','I-MONEY','I-MONEY','I-MONEY','I-MONEY','O','O','O','O','O','B-DATE','I-DATE','I-DATE','I-DATE','O'],
            ['B-ORG','I-ORG','I-ORG','O','O','O','O','O','O','O','O',"B-ORG",'O'],
            ['O','O','B-DATE','O','B-ORG','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','B-LOC','O','O','B-LOC','I-LOC','O','B-LOC','O','B-LOC','O','B-LOC','O','O','O','O'],
            ['B-ORG','O','O','B-MONEY','I-MONEY','I-MONEY','I-MONEY','I-MONEY','I-MONEY','O','O','O','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','O','B-ATTR','I-ATTR','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','B-LOC','O','B-LOC','B-LOC','O','B-LOC','O','O'],
            ['O','B-LOAN','I-LOAN','I-LOAN','I-LOAN','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','O','B-ATTR','I-ATTR','I-ATTR','O','O','O','O','O','O','O'],
            ['O','B-LOAN','I-LOAN','I-LOAN','I-LOAN','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O'],
            ['O','B-LOAN','I-LOAN','I-LOAN','I-LOAN','O','O','O','O','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','O','O','O','O','O','O','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','O','O','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','B-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','I-ATTR','O','O','O','O','O','O','O']
        ]
        self.build_dataset()
        
    def build_dataset(self):

        unique_lables = set()
        for tag in self.tags:
            unique_lables = unique_lables | set(tag)
        label_list = sorted(list(unique_lables))
        self.tags_len = len(label_list)
        self.tag2id = {tag:idx for idx, tag in enumerate(label_list)}
        self.id2tag = {idx:tag for idx, tag in enumerate(label_list)}
            
        word_ids = [glove_vocab(sentence) for sentence in self.sentences]
        tag_ids = [[self.tag2id[tag] for tag in tags] for tags in self.tags]
        for idx in range(len(word_ids)):
            self.samples.append({
                'sample_id': idx,
                'input_ids': word_ids[idx],
                'tag_ids': tag_ids[idx],
                'seq_len': len(tag_ids[idx])
            })

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    sample_id = [sample['sample_id'] for sample in batch]
    input_ids = [torch.LongTensor(sample['input_ids']) for sample in batch]
    tag_ids = [torch.LongTensor(sample['tag_ids']) for sample in batch]
    seq_len = [sample['seq_len'] for sample in batch]
    pad_input_ids = pad_sequence(input_ids, batch_first=True)
    pad_tag_ids = pad_sequence(tag_ids)
    return  {
        'sample_id': sample_id,
        'input_ids': pad_input_ids,
        'tag_ids': pad_tag_ids,
        'seq_len': seq_len
    }

if __name__ == "__main__":
    dataset = ExampleDataset()
    dataloder = DataLoader(dataset,batch_size = 2, collate_fn=collate_fn, shuffle=True)
    embedding_dim = vectors.shape[1]
    lstm_hidden_dim = 100
    embeddings = torch.nn.Embedding(len(vectors), embedding_dim)
    lstm = torch.nn.LSTM(embedding_dim, lstm_hidden_dim,bidirectional=True)
    linear = torch.nn.Linear(200, dataset.tags_len)
    loss_fn = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=-1)
    for idx, batch in enumerate(dataloder):
        print(idx)
        embd = embeddings(batch['input_ids'])
        embd = pack_padded_sequence(embd, batch['seq_len'], batch_first=True,enforce_sorted=False)
        outputs,_ = lstm(embd)
        encoding, lens = pad_packed_sequence(outputs)
        logits = linear(encoding)
        tags = batch['tag_ids']
        mask = (tags != 0) & (tags != -100)
        logits_no_pad, tags_no_pad = logits[mask], tags[mask]
        pred = softmax(logits)
        pred = torch.argmax(pred, dim=-1)
        pred_no_pad, labels_no_pad = pred[mask], tags[mask]
        f1 = f1_score(pred_no_pad, labels_no_pad, num_classes=dataset.tags_len, average="macro")
        loss = loss_fn(logits.view(-1, logits.shape[-1]), tags.view(-1))
        print(f1, loss)
