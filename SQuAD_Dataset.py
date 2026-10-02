import torch
import json
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TDataset
from data_preprocess import load_data,train_data_preprocess,preprocess_validation_examples
from transformers import AutoTokenizer
with open("config.json") as file:
        conf = json.load(file)
trained_checkpoint = conf["checkpoint"]
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)

class SQuAD(TDataset):
    def __init__(self,mode,Data_Path,tokenizer_max_length=512,tokenizer_stride=128):
        self.mode = mode
        if self.mode == "train":
            df = load_data(Data_Path)
            self.sample = HFDataset.from_pandas(df)
            print("Before tokenization length")
            print(len(self.sample))
            self.dataset = self.sample.map(
                train_data_preprocess,
                batched=True,
                remove_columns=self.sample.column_names,
                fn_kwargs={"tokenizer": tokenizer,"tokenizer_max_length": tokenizer_max_length, "tokenizer_stride":tokenizer_stride}
            )
            print("After tokenization length")
            print(len(self.dataset))
        
        elif self.mode == "val":
            df = load_data(Data_Path)
            self.sample = HFDataset.from_pandas(df)
            print("Before tokenization length")
            print(len(self.sample))
            self.dataset = self.sample.map(
                preprocess_validation_examples,
                batched=True,
                remove_columns=self.sample.column_names,
                fn_kwargs={"tokenizer": tokenizer,"tokenizer_max_length": tokenizer_max_length, "tokenizer_stride":tokenizer_stride}
            )
            print("After tokenization length")
            print(len(self.dataset))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        out = {}
        example = self.dataset[idx]
        out['input_ids'] = torch.tensor(example['input_ids'])
        out['attention_mask'] = torch.tensor(example['attention_mask'])
        
        if self.mode == "train":
            out['start_positions'] = torch.unsqueeze(torch.tensor(example['start_positions']),dim=0)
            out['end_positions'] = torch.unsqueeze(torch.tensor(example['end_positions']),dim=0)
        
        return out