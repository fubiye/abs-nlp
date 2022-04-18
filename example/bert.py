from transformers import BertTokenizer, BertModel, AutoTokenizer
import os

# # tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser("~/.cache/bert"))
# tokenizer = BertTokenizer.from_pretrained(r'D:\dataset\fin-bert\FinBERT-FinVocab-Uncased')
# sequence = ["In a hole in the ground there lived a hobbit."]
sequence = "DBJPM 2017 C-12"
# tokenized_input = tokenizer(sequence)
# # tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
# print(tokenized_input['input_ids'])
# model = BertModel.from_pretrained(r'D:\dataset\fin-bert\FinBERT-Prime_128MSL-250K')
# # model = AutoModelForSequenceClassification.from_pretrained(r'D:\dataset\fin-bert\FinBERT-FinVocab-Uncased')
# outputs = model(tokens)
# print(outputs)


# from transformers import BertTokenizer, BertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
text = ["DBJPM 2017 C-12","hello world"]

words = [["DBJPM","2017","C-12"],["hello", "world"]]
tags = [[1,2,3],[3,1]]
# tokenized_inputs  = tokenizer(words, padding=True, truncation=True,is_split_into_words=True)

sentence = ["DBJPM","2017","C-12"]
tokenized_inputs  = tokenizer(sentence, is_split_into_words=True)
print(tokenized_inputs)
word_ids = tokenized_inputs.word_ids()
print(word_ids)

# labels = []

for i, label in enumerate(tags):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(label[word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx
    labels.append(label_ids)
# tokenized_inputs["labels"] = labels

# output = model(**encoded_input)
# print(output[0].shape)