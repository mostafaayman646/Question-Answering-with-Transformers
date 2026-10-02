from transformers import default_data_collator,DistilBertForQuestionAnswering
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
import json
from SQuAD_Dataset import SQuAD
from trainer import train
from utils import dev

########################################################1. Config ##############################################################
with open("config.json") as file:
        conf = json.load(file)

############################################1. Creating Dataset class ##########################################################
Train_Path = conf["Train_Path"]
Train_Dataset = SQuAD(mode="train",Data_Path=Train_Path,tokenizer_max_length=conf["tokenizer_max_length"],tokenizer_stride=conf["tokenizer_stride"])

Val_Path = conf["Val_Path"]
Val_Dataset = SQuAD(mode="val",Data_Path=Val_Path,tokenizer_max_length=conf["tokenizer_max_length"],tokenizer_stride=conf["tokenizer_stride"])

for i,d in enumerate(Train_Dataset):
    for k in d.keys():
        print(k + ' : ', d[k].shape)
    print('--'*40)

    if i == 3:
        break
        
print('__'*50)

for i,d in enumerate(Val_Dataset):
    for k in d.keys():
        print(k + ' : ', len(d[k]))
    print('--'*40)
    
    if i == 3:
        break

######################################################2. DataLoader ############################################################
BATCH_SIZE = conf["BATCH_SIZE"]
train_dataloader = DataLoader(
    Train_Dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=BATCH_SIZE,
)
eval_dataloader = DataLoader(
    Val_Dataset, collate_fn=default_data_collator,
    batch_size=BATCH_SIZE
)

for batch in train_dataloader:
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['start_positions'].shape)
    print(batch['end_positions'].shape)
    break

print('---'*20)

for batch in eval_dataloader:
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    break

######################################################3. Train #################################################################
device = dev(conf["device"])
print(f'Available device: {device}')
model = DistilBertForQuestionAnswering.from_pretrained(conf["checkpoint"])
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=conf["lr"])
train(device=device,model=model,optimizer=optimizer,seed_val=conf["seed_val"],train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,val_processed = Val_Dataset.dataset,val = Val_Dataset.sample,
        epochs = conf["epochs"])

######################################################4. Save Model ############################################################
model_save_path = 'fine_tuned_model.pt'
torch.save(model.state_dict(), model_save_path)