import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from utils import preprocess_sentence,LanguageIndex,max_length , predict_sentences,generate_sentence , clean_tags
from Dataset import MyData
from sentence_transformers import SentenceTransformer
from Models import Encoder2,Decoder2

data = pd.read_csv('data/all_train.csv')
data["QA"] = data.QA.apply(lambda w: preprocess_sentence(w))
data["TAG"] = data.TAG.apply(lambda w: preprocess_sentence(w))
data.sample(10)

### Encode sentences
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
data["QA"] = np.load('embeddings.npy', allow_pickle=True)
print('loaded')
data.QA.iloc[0].shape

### Encode only targets
targ_lang = LanguageIndex(data["TAG"].values.tolist())
target_tensor = [[targ_lang.word2idx[s] for s in eng.split(' ')] for eng in data["TAG"].values.tolist()]
max_length_tar = max_length(target_tensor)
target_tensor = pad_sequences(target_tensor, max_length_tar)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(data.QA, target_tensor,
                                                                                        shuffle=True,
                                                                                                test_size=0.2)
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

BATCH_SIZE = 64
embedding_dim = 384
units = 1024
vocab_tar_size = len(targ_lang.word2idx)
train_dataset = MyData(data.QA.values, target_tensor)
dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                     drop_last=True,
                     shuffle=True)


criterion = nn.CrossEntropyLoss()

def loss_function(real, pred):
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = Encoder2(embedding_dim, units, BATCH_SIZE)
decoder = Decoder2(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE)

encoder.to(device)
decoder.to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=0.001)

EPOCHS = 4
encoder.batch_sz = 64
encoder.initialize_hidden_state(device)
decoder.batch_sz = 64
decoder.initialize_hidden_state()

for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    total_loss = 0

    for (batch, (inp, targ, inp_len)) in enumerate(dataset):
        loss = 0
        enc_output = encoder(inp.to(device), device)
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
        for t in range(1, targ.size(1)):
            predictions, dec_hidden = decoder(dec_input.to(device),
                                              enc_output.to(device))

            loss += loss_function(targ[:, t].long().to(device), predictions.to(device))
            dec_input = targ[:, t].unsqueeze(1)

        batch_loss = (loss / int(targ.size(1)))
        total_loss += batch_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.detach().item()))

encoder.batch_sz = 1
encoder.initialize_hidden_state(device)
decoder.batch_sz = 1
decoder.initialize_hidden_state()

test_sentence = " which films have the same screenwriter of a tree grows in brooklyn"
test_sentence = [sbert.encode(test_sentence)]
ret = generate_sentence(encoder, decoder, torch.tensor(test_sentence), device, targ_lang, max_length=max_length_tar)
print(ret)

data_test = pd.read_csv('data/all_test.csv').dropna().sample(12000)
tags_pred = predict_sentences(data_test['QA'].values,sbert,encoder,decoder)
tags_true = [preprocess_sentence(sentence).split(' ') for sentence in data_test['TAG']]
tags_original = tags_pred.copy()
true_tags_original = tags_true.copy()

new_tags_pred = [clean_tags(tags) for tags in tags_original]
new_tags_true = [clean_tags(tags) for tags in true_tags_original]

tags_true_processed = np.array([' '.join(words) for words in new_tags_pred])
tags_pred_processed = np.array([' '.join(words) for words in new_tags_true])
print(tags_pred_processed.shape, tags_true_processed.shape)

results = pd.DataFrame(np.array([data_test.QA.values, tags_true_processed, tags_pred_processed]).transpose(),
                       columns=['QA', 'Original', 'Predicted'])
results.to_csv('results_sbert.csv')
