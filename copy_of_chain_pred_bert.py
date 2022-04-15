import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from keras.preprocessing.sequence import pad_sequences 
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import nltk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

data = pd.read_csv('parttrain.csv')
data.head(5)

# Converts the unicode file to ascii
def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z0-9_?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

"""## Data Exploration
Let's explore the dataset a bit.
"""

# Now we do the preprocessing using pandas and lambdas
data["QA"] = data.QA.apply(lambda w: preprocess_sentence(w))
data["TAG"] = data.TAG.apply(lambda w: preprocess_sentence(w))
data.sample(10)

"""#### Building Vocabulary Index

"""

class LanguageIndex():
    def __init__(self, lang):
        """ lang are the list of phrases from each language"""
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for phrase in self.lang:
            # update with individual tokens
            self.vocab.update(phrase.split(' '))
        # sort the vocab
        self.vocab = sorted(self.vocab)
        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        # word to index mapping
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
input_tensor[:10]

target_tensor[:10]

def max_length(tensor):
    return max(len(t) for t in tensor)
# calculate the max_length of input and output tensor
max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

input_tensor = pad_sequences(input_tensor, max_length_inp)
target_tensor = pad_sequences(target_tensor, max_length_tar)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor,shuffle=True, test_size=0.2)
# Show length
len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

"""## Load data into DataLoader for Batching
This is just preparing the dataset so that it can be efficiently fed into the model through batches.
"""

from torch.utils.data import Dataset, DataLoader

# convert the data to tensors and pass to the Dataloader 
# to create an batch iterator

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x,y,x_len
    
    def __len__(self):
        return len(self.data)

"""## Parameters
Let's define the hyperparameters and other things we need for training our NMT model.
"""

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

train_dataset = MyData(input_tensor_train, target_tensor_train)
val_dataset = MyData(input_tensor_val, target_tensor_val)

dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)

val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                     drop_last=True,
                     shuffle=True)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.enc_units)
        
    def forward(self, x, device):
        # x: batch_size, max_length 
        
        # x: batch_size, max_length, embedding_dim
        x = self.embedding(x) 
                
        # x transformed = max_len X batch_size X embedding_dim
        # x = x.permute(1,0,2)
        #x = pack_padded_sequence(x, lens) # unpad
    
        self.hidden = self.initialize_hidden_state(device)
        
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        # pad the sequence to the max length in the batch
        #output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)

"""### Decoder

Here, we'll implement an encoder-decoder model with attention which you can read about in the TensorFlow [Neural Machine Translation (seq2seq) tutorial](https://github.com/tensorflow/nmt). This example uses a more recent set of APIs. This notebook implements the [attention equations](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism) from the seq2seq tutorial. The following diagram shows that each input word is assigned a weight by the attention mechanism which is then used by the decoder to predict the next word in the sentence.

<img src="https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg" width="500" alt="attention mechanism">

The input is put through an encoder model which gives us the encoder output of shape *(batch_size, max_length, hidden_size)* and the encoder hidden state of shape *(batch_size, hidden_size)*. 

Here are the equations that are implemented:

<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_0.jpg" alt="attention equation 0" width="800">
<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">

We're using *Bahdanau attention*. Lets decide on notation before writing the simplified form:

* FC = Fully connected (dense) layer
* EO = Encoder output
* H = hidden state
* X = input to the decoder

And the pseudo-code:

* `score = FC(tanh(FC(EO) + FC(H)))`
* `attention weights = softmax(score, axis = 1)`. Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*, since the shape of score is *(batch_size, max_length, 1)*. `Max_length` is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.
* `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
* `embedding output` = The input to the decoder X is passed through an embedding layer.
* `merged vector = concat(embedding output, context vector)`
* This merged vector is then given to the GRU
  
The shapes of all the vectors at each step have been specified in the comments in the code:
"""

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units, 
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.enc_units, self.vocab_size)
        
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
    
    def forward(self, x, hidden, enc_output):
        enc_output = enc_output.permute(1,0,2)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        output, state = self.gru(x)
        output =  output.view(-1, output.size(2))
        x = self.fc(output)
        return x, state, attention_weights
    
    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))

criterion = nn.CrossEntropyLoss()
def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
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

"""## Training
Now we start the training. We are only using 10 epochs but you can expand this to keep trainining the model for a longer period of time. Note that in this case we are teacher forcing during training. Find a more detailed explanation in the official TensorFlow [implementation](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb) of this notebook provided by the TensorFlow team. 

- Pass the input through the encoder which return encoder output and the encoder hidden state.
- The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.
- The decoder returns the predictions and the decoder hidden state.
- The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
- Use teacher forcing to decide the next input to the decoder.
- Teacher forcing is the technique where the target word is passed as the next input to the decoder.
- The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
"""

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
        # use teacher forcing - feeding the target as the next input (via dec_input)
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * 1)
        # run code below for every timestep in the ys batch
        out_sentence = []
        for t in range(1, sentence.size(0)):
            predictions, dec_hidden, _ = decoder(dec_input.to(device), 
                                        dec_hidden.to(device), 
                                        enc_output.to(device))
            dec_input = predictions.argmax(dim=1).unsqueeze(1)
            # print(dec_input)
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
            #loss += loss_
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

def translate_sentence(encoder, decoder, sentence, max_length=120):
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
ret = translate_sentence(encoder, decoder, torch.tensor(test_sentence), max_length=max_length_tar)
ret

from torchtext.data.metrics import bleu_score

def predict_sentences(sentences):
    def predict_sentence(test_sentence):
        test_sentence = preprocess_sentence(test_sentence)
        test_sentence = [[inp_lang.word2idx[s] for s in test_sentence.split(' ') if s in inp_lang.word2idx]]
        test_sentence = pad_sequences(test_sentence, max_length_inp)
        return translate_sentence(encoder, decoder, torch.tensor(test_sentence), max_length=max_length_tar)

    return [predict_sentence(sentence) for sentence in sentences]


test_sentences = ["which films the .", "which films the .", ]

data_test = pd.read_csv('all_test.csv')

tags_pred = predict_sentences(data_test['QA'].values)
tags_true = [preprocess_sentence(sentence).split() for sentence in data_test['TAG']]

print(bleu_score(tags_pred, tags_true))

