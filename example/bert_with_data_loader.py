from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
# tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[2])
# word_ids = tokenized_inputs.word_ids(batch_index=2)
# for (token, word_id) in zip(tokens, word_ids):
#     print(f'{token}\t{word_id}')

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
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.build_dataset()
        
    def build_dataset(self):
        
        unique_lables = set()
        for tag in self.tags:
            unique_lables = unique_lables | set(tag)
        label_list = sorted(list(unique_lables))
        self.tags_len = len(label_list)
        self.tag2id = {tag:idx for idx, tag in enumerate(label_list)}
        self.id2tag = {idx:tag for idx, tag in enumerate(label_list)}
            
        tokenized_inputs = self.tokenizer(self.sentences, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
                
        for idx, (input_ids,token_type_ids,attention_mask) in enumerate(zip(tokenized_inputs['input_ids'],tokenized_inputs['token_type_ids'],tokenized_inputs['attention_mask'])):
            tags = self.tags[idx]
            tag_ids = [self.tag2id[tag] for tag in tags]
            word_ids = tokenized_inputs.word_ids(batch_index=idx)
            previous_word_id = -1
            token_tag_ids = []
            for word_id in word_ids:
                if word_id is None:
                    token_tag_ids.append(0)
                elif word_id == previous_word_id:
                    token_tag_ids.append(-1000)
                else:
                    token_tag_ids.append(tag_ids[word_id])
            self.samples.append({
                'sample_id': idx,
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'tag_ids': torch.LongTensor(token_tag_ids)
            })
        # for idx, (sentence, tags) in enumerate(zip(self.sentences, self.tags)):
        #     self.samples.append({
        #         'sample_id': idx,
        #         'words': sentence,
        #         'tags': tags
        #     })
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    model = AutoModel.from_pretrained('bert-base-uncased')
    dataset = ExampleDataset()
    dataloder = DataLoader(dataset,batch_size = 2,shuffle=True)
    linear = torch.nn.Linear(768, dataset.tags_len)
    loss_fn = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=-1)
    for idx, batch in enumerate(dataloder):
        print(idx)
        outputs = model(input_ids=batch['input_ids'], token_type_ids=batch['token_type_ids'],attention_mask=batch['attention_mask'])
        seq_outputs = outputs[0]
        logits = linear(seq_outputs)
        tags = batch['tag_ids']
        mask = (tags != 0) & (tags != -1000)
        logits_no_pad, tags_no_pad = logits[mask], tags[mask]
        # pred = softmax(logits)
        # pred = torch.argmax(pred, dim=-1)
        # pred_no_pad, labels_no_pad = pred[mask], labels[mask]
        # f1 = f1_score(pred_no_pad, labels_no_pad, num_classes=self.num_labels, average="macro")
        loss = loss_fn(logits_no_pad.view(-1, logits_no_pad.shape[-1]), tags_no_pad.view(-1))
        print(loss)
