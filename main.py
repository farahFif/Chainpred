from Dataprep import Data_prep
from sentence_transformers import SentenceTransformer
from collections import Counter
import numpy as np
from Dataset import Data_torch_loader
from torch.utils.data import DataLoader
import pandas as pd


#### get and prepare  data
d = Data_prep('MetaQA')
d.train.iloc[:5000].to_csv('parttrain.csv')
exit()
d.fix_data()

train_tags = d.train['TAG']
d.prep_for_train()
d.prep_for_valid()
d.prep_for_test()


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
#vocab_inp_size = len(inp_lang.word2idx)
#vocab_tar_size = len(targ_lang.word2idx)


train_dataset = Data_torch_loader(input_tensor_train, target_tensor_train)
val_dataset = Data_torch_loader(input_tensor_val, target_tensor_val)

dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE,
                     drop_last=True,
                     shuffle=True)

val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE,
                     drop_last=True,
                     shuffle=True)