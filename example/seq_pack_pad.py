import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence, pack_padded_sequence

text1 = torch.tensor([1,2,3,4])
text2 = torch.tensor([5,6,7])
text3 = torch.tensor([8,9])

sequences = [text1, text2, text3]

packed_seq = pack_sequence(sequences)
print(packed_seq)

