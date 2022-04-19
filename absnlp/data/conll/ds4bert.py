from typing import Dict, List

import hydra
import torch
from torch.utils.data.dataset import Dataset, T_co
from absnlp.vocab.customized import Vocab
from transformers import AutoTokenizer
class CoNLLDataset4Bert(Dataset):
    def __init__(self,padding_size: int, dataset_path: str, vocabs:dict=None):
        self.words: List[List[str]] = []
        self.ner_labels: List[List[str]] = []
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.parse_dataset(dataset_path)
        self.len_dataset: int = len(self.words)
        self.vocab_words: Vocab = self.build_vocab(self.words) if vocabs is None else vocabs["words"]
        self.vocab_label_ner: Vocab = self.build_vocab(self.ner_labels, is_label=True) if vocabs is None else vocabs["ner_labels"]
        self.padding_size = padding_size
        self.dataset = self.build_dataset()

    def get_vocab_sizes(self) -> Dict[str, int]:
        vocab_words_size = len(self.vocab_words)
        vocab_ner_labels_size = len(self.vocab_label_ner)
        return {"words": vocab_words_size, "ner_labels": vocab_ner_labels_size}

    def build_dataset(self) -> List[dict]:
        result = []
        for id_sentence in range(self.len_dataset):
            sentence = self.words[id_sentence]
            tokenized_inputs  = self.tokenizer(sentence, is_split_into_words=True)
            encoded_ner_labels = self.encode_tokens(self.ner_labels[id_sentence], self.vocab_label_ner)
            self.assign_token_tags(tokenized_inputs, encoded_ner_labels)
            sample = {
                "id_sentence": id_sentence,
                "input_ids": tokenized_inputs['input_ids'],
                "labels": tokenized_inputs['labels'],
            }
            result.append(sample)
        return result

    def assign_token_tags(self, tokenized_inputs, word_labels):
        previous_word_idx = None
        label_ids = []
        word_ids = tokenized_inputs.word_ids()
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(word_labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        input_ids = tokenized_inputs['input_ids']
        input_ids_len = len(input_ids)
        if input_ids_len < self.padding_size:
            padding: List[int] = [0] * (self.padding_size - input_ids_len)
            input_ids.extend(padding)
            label_ids.extend(padding)
        tokenized_inputs['labels'] = torch.LongTensor(label_ids)
        
    def __getitem__(self, index) -> T_co:
        return self.dataset[index]

    def __len__(self):
        return self.len_dataset

    def build_vocab(
        self,
        token_list: List[List[str]],
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        freq_to_drop: int = 0,
        is_label: bool = False,
    ) -> Vocab:
        """
        build a vocabulary for the roles label vector
        """
        vocab = Vocab(pad_token=pad_token, unk_token=unk_token, is_label=is_label)
        for sentence_id in range(self.len_dataset):
            for token in token_list[sentence_id]:
                vocab.add_token(token)
        vocab.drop_frequency(freq_to_drop=freq_to_drop)
        return vocab

    def encode_tokens(self, sentence: List[str], vocab: Vocab) -> torch.LongTensor:
        result: List[int] = [vocab.token_to_id(token) if token in vocab else vocab.unk_id for token in sentence]
        # len_sentence = len(sentence)
        # if len_sentence < self.padding_size:
        #     padding: List[int] = [vocab.pad_id] * (self.padding_size - len_sentence)
        #     result.extend(padding)
        return torch.LongTensor(result)

    def parse_dataset(self, dataset_path: str) -> None:
        with open(dataset_path) as f:
            next(f)
            next(f)
            raw_text = f.read()
            sentences: List[str] = raw_text.split("\n\n")
            sentences: List[List[str]] = [sentence.split("\n") for sentence in sentences][:-1]
            # sentence:List[List[List[str]]] = [row.split(" ") for sentence in sentences for row in sentence]

            for sent in sentences:
                words = []
                ner_labels = []
                for row in sent:
                    (word, _, _, ner_label) = row.split(" ")
                    words.append(word)
                    ner_labels.append(ner_label)

                self.words.append(words)
                self.ner_labels.append(ner_labels)

    def get_vocabs(self) -> dict:
        return {"words": self.vocab_words, "ner_labels": self.vocab_label_ner}
