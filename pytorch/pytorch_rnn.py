import torch
import time

rnn = torch.nn.RNN(150528, 32, 50)
input = torch.randn(20, 1, 150528)
h0 = torch.randn(50, 1, 32)

t = time.time()
output, hn = rnn(input, h0)
print('pytorch rnn: ', (time.time()-t)*1000)

print (output.shape)
torch.onnx.export(rnn, input, "pytorch_rnn.onnx")