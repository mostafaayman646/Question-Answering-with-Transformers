from transformers import default_data_collator
from torch.utils.data import DataLoader

from SQuAD_Dataset import SQuAD

############################################1. Creating Dataset class ##########################################################
Train_Path = "data/train-v2.0.json"
Train_Dataset = SQuAD(mode="train",Data_Path=Train_Path,tokenizer_max_length=512,tokenizer_stride=128)

Val_Path = "data/dev-v2.0.json"
Val_Dataset = SQuAD(mode="val",Data_Path=Val_Path,tokenizer_max_length=512,tokenizer_stride=128)

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
################################################################################################################################
######################################################2. DataLoader ############################################################
BATCH_SIZE = 2
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