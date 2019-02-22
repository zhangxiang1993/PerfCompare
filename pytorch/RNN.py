from torch.autograd import Variable
import torch.onnx
import torchvision
import torch.nn as nn
import numpy

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
#         self.rnn = nn.LSTM(ninp, nhid, num_layers=nlayers, dropout=dropout)
        self.rnn = nn.RNN(ninp, nhid, num_layers=nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()
        self.nlayers = nlayers
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = emb
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_())
        # return (Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()),
        #        Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()))

model = RNNModel(ntoken=64, ninp=128, nhid=128, nlayers=2, dropout=0.5)

dummy_input = Variable(torch.from_numpy(numpy.arange(256)[:, None] % 64))
dummy_state = model.init_hidden(batch_size=1)

# works well
output = model(dummy_input, dummy_state)[0].size()

# torch.onnx.export(model, (dummy_input, dummy_state), "recurrent.proto", verbose=True)