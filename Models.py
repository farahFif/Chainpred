import torch

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
        x = self.embedding(x)
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(x, self.hidden)
        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)


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
        enc_output = enc_output.permute(1, 0, 2)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        attention_weights = torch.softmax(self.V(score), dim=1)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        x = self.embedding(x)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        output, state = self.gru(x)
        output = output.view(-1, output.size(2))
        x = self.fc(output)
        return x, state, attention_weights

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))
