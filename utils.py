import numpy as np
import pandas as pd
import unicodedata
import re
import torch
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
    w = re.sub(r"[^a-zA-Z0-9_?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

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
            self.word2idx[word] = index + 1  # +1 because of pad token
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0, 1), y, lengths  # transpose (batch x seq) to (seq x batch)


def generate_sentence(encoder, decoder, sentence, device, targ_lang, max_length=120):
    encoder.eval()
    decoder.eval()
    sentence = sentence.view(1, -1)
    with torch.no_grad():
        enc_output = encoder(sentence.to(device), device)
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * 1)
        out_sentence = []
        next_word = '<start>'
        while next_word != '<end>':
            predictions, dec_hidden = decoder(dec_input.to(device),
                                              enc_output.to(device))
            dec_input = predictions.argmax(dim=1).unsqueeze(1)
            next_word = targ_lang.idx2word[predictions.squeeze().argmax().item()]
            out_sentence.append(next_word)
            if len(out_sentence) > max_length:
                break
    return out_sentence

def generate_sentence2(encoder, decoder, sentence, device, targ_lang, max_length=120):
    encoder.eval()
    decoder.eval()
    sentence = sentence.view(1, -1)
    with torch.no_grad():
        enc_output , hid = encoder(sentence.to(device), device)
        dec_hid = hid
        dec_input = torch.tensor([[targ_lang.word2idx['<start>']]] * 1)
        out_sentence = []
        next_word = '<start>'
        while next_word != '<end>':
            predictions, dec_hidden,_ = decoder(dec_input.to(device), dec_hid.to(device),
                                         enc_output.to(device))
            dec_input = predictions.argmax(dim=1).unsqueeze(1)
            next_word = targ_lang.idx2word[predictions.squeeze().argmax().item()]
            out_sentence.append(next_word)
            if len(out_sentence) > max_length:
                break
    return out_sentence


def predict_sentences(sentences,sbert,encoder,decoder,max_length_tar,targ_lang,device,method=1):

    def predict_sentence(test_sentence,k):
        print("here",k)
        test_sentence = preprocess_sentence(test_sentence)
        if sbert != None:
            test_sentence = np.array([sbert.encode(test_sentence)])
        if method == 2:
            return generate_sentence2(encoder, decoder, torch.tensor(test_sentence), device,targ_lang,max_length=max_length_tar)
        else:
            return generate_sentence(encoder, decoder, torch.tensor(test_sentence), device,targ_lang,max_length=max_length_tar)
    return [predict_sentence(sentence,k) for k,sentence in enumerate(sentences)]

def clean_tags(tags):
    while len(tags) > 0 and tags[0].startswith("<"):
        tags.pop(0)
    if '<end>' in tags:
        return tags[:tags.index('<end>')]
    return [tag for tag in tags if not tag.startswith("<")]

def calc_score(path):
    data = pd.read_csv(path).dropna()
    data['Original'] = data.Original.apply(lambda x : x.split(' '))
    data['Predicted'] = data.Predicted.apply(lambda x : x.split(' '))
    score = 0
    for index ,x in data.iterrows():
        if x['Original'] == x['Predicted']:
            score +=1
    return score/len(data)



h = calc_score('results_sbert_meth2.csv')
print(h)