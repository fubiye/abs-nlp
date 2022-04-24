import torch
from .bilstm_ner import BiLstmSoftmaxModel

class ModelLoader():
    
    @classmethod
    def from_pretrained(cls,args, path):
        model = BiLstmSoftmaxModel(args)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model