from torch.utils.data import DataLoader
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from keras.preprocessing.sequence import pad_sequences 
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from Models import Encoder , Decoder
import unicodedata
import re
import nltk
from torchtext.data.metrics import bleu_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from Dataset import MyData


data = pd.read_csv('all_train.csv')
valid = pd.read_csv('all_valid.csv')
data.head(5)

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9_?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

data["QA"] = data.QA.apply(lambda w: preprocess_sentence(w))
data["TAG"] = data.TAG.apply(lambda w: preprocess_sentence(w))


valid["QA"] = valid.QA.apply(lambda w: preprocess_sentence(w))
valid["TAG"] = valid.TAG.apply(lambda w: preprocess_sentence(w))

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word      


# index language using the class above
inp_lang = LanguageIndex(data["QA"].values.tolist())
targ_lang = LanguageIndex(data["TAG"].values.tolist())
# Vectorize the input and target languages
input_tensor = [[inp_lang.word2idx[s] for s in es.split(' ')]  for es in data["QA"].values.tolist()]
target_tensor = [[targ_lang.word2idx[s] for s in eng.split(' ')]  for eng in data["TAG"].values.tolist()]

# index language using the class above
inp_lang_valid = LanguageIndex(valid["QA"].values.tolist())
targ_lang_valid = LanguageIndex(valid["TAG"].values.tolist())
# Vectorize the input and target languages
input_tensor_valid = [[inp_lang.word2idx[s] for s in p.split(' ')]  for p in valid["QA"].values.tolist()]
target_tensor_valid = [[targ_lang.word2idx[s] for s in p.split(' ')]  for p in valid["TAG"].values.tolist()]


def max_length(tensor):
    return max(len(t) for t in tensor)

# calculate the max_length of input and output tensor
max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
max_length_inp_valid, max_length_tar_valid = max_length(input_tensor_valid), max_length(target_tensor_valid)


input_tensor = pad_sequences(input_tensor, max_length_inp)
target_tensor = pad_sequences(target_tensor, max_length_tar)


input_tensor_valid = pad_sequences(input_tensor_valid, max_length_inp)
target_tensor_valid = pad_sequences(target_tensor_valid, max_length_tar)


input_tensor_train, target_tensor_train =  input_tensor , target_tensor
input_tensor_valid , target_tensor_valid = input_tensor_valid ,target_tensor_valid

print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_valid), len(target_tensor_valid))


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

train_dataset = MyData(input_tensor_train, target_tensor_train)
val_dataset = MyData(input_tensor_valid, target_tensor_valid)

dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)

val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)

criterion = nn.CrossEntropyLoss()
def loss_function(real, pred):
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, units, BATCH_SIZE)

encoder.to(device)
decoder.to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                       lr=0.001)


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)

EPOCHS = 1
def eval2(encoder, decoder, sentence, max_length=120):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    sentence = torch.unsqueeze(sentence, dim=1)
    with torch.no_grad():
        print(sentence.size())
        enc_output, enc_hidden = encoder(sentence.to(device), [sentence.size(0)], device)
        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * 1)
        out_sentence = []
        for t in range(1, sentence.size(0)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                        dec_hidden.to(device), 
                                        enc_output.to(device))
            dec_input = predictions.argmax(dim=1).unsqueeze(1)
            out_sentence.append(targ_lang.idx2word[predictions.squeeze().argmax().item()])
            # print(out_sentence)
            # print(predictions.size())
    return out_sentence


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
        xs, ys, lens = sort_batch(inp, targ, inp_len)
        enc_output, enc_hidden = encoder(xs.to(device), device)
        dec_hidden = enc_hidden
        # use teacher forcing - feeding the target as the next input (via dec_input)
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * BATCH_SIZE)
        # run code below for every timestep in the ys batch
        for t in range(1, ys.size(1)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                         dec_hidden.to(device), 
                                         enc_output.to(device))
            
            loss += loss_function(ys[:, t].long().to(device), predictions.to(device))
            dec_input = ys[:, t].unsqueeze(1)

        batch_loss = (loss / int(ys.size(1)))
        total_loss += batch_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.detach().item()))

def gen_sentence(encoder, decoder, sentence, max_length=120):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    sentence = sentence.transpose(0,1) 
    with torch.no_grad():
        enc_output, enc_hidden = encoder(sentence.to(device),device)
        dec_hidden = enc_hidden
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * 1)
        out_sentence = []
        for t in range(1, sentence.size(0)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                        dec_hidden.to(device), 
                                        enc_output.to(device))
            dec_input = predictions.argmax(dim=1).unsqueeze(1)
            out_sentence.append(targ_lang.idx2word[predictions.squeeze().argmax().item()])

    return out_sentence

encoder.batch_sz = 1
encoder.initialize_hidden_state(device)
decoder.batch_sz = 1
decoder.initialize_hidden_state()

test_sentence = "<start> which films have the same screenwriter of a tree grows in brooklyn . <end>"
test_sentence = [[inp_lang.word2idx[s] for s in test_sentence.split(' ')]]
test_sentence = pad_sequences(test_sentence, max_length_inp)
ret = gen_sentence(encoder, decoder, torch.tensor(test_sentence), max_length=max_length_tar)
ret


def predict_sentences(sentences):
    def predict_sentence(test_sentence):
        test_sentence = preprocess_sentence(test_sentence)
        test_sentence = [[inp_lang.word2idx[s] for s in test_sentence.split(' ') if s in inp_lang.word2idx]]
        test_sentence = pad_sequences(test_sentence, max_length_inp)
        return gen_sentence(encoder, decoder, torch.tensor(test_sentence), max_length=max_length_tar)

    return [predict_sentence(sentence) for sentence in sentences]


test_sentences = ["which films the .", "which films the .", ]

data_test = pd.read_csv('all_test.csv')

tags_pred = predict_sentences(data_test['QA'].values)
tags_true = [preprocess_sentence(sentence).split() for sentence in data_test['TAG']]

print(bleu_score(tags_pred, tags_true))

data_test['TAG'] = data_test['TAG'].fillna('')
tags_true = [preprocess_sentence(sentence).split() for sentence in data_test['TAG'].values]

print(tags_pred[0], tags_true[0])

tags_original = tags_pred.copy()
true_tags_original = tags_true.copy()
def clean_tags(tags):
    while len(tags) > 0 and tags[0].startswith("<"):
        tags.pop(0)
    if '<end>' in tags:
        return tags[:tags.index('<end>')]
    return [tag for tag in tags if not tag.startswith("<")]
new_tags_pred = [clean_tags(tags) for tags in tags_original]
new_tags_true = [clean_tags(tags) for tags in true_tags_original]
print(new_tags_pred[0], new_tags_true[0])

print(new_tags_true.pop(9947))
print(new_tags_pred.pop(9947))

print(bleu_score(new_tags_pred, new_tags_true, max_n=1, weights=[1]))

print(new_tags_true[:-10])
print(new_tags_pred[:-10])

