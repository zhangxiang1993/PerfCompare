import torch
import time

rnn = torch.nn.RNN(150528, 32, 1)

t = time.time()
for i in range(100):
    input = torch.randn(1, 1, 150528)
    h0 = torch.randn(1, 1, 32)
    output, hn = rnn(input, h0)
print('pytorch rnn: ', (time.time()-t)*100) # 140.5630111694336

print (output.shape)
torch.onnx.export(rnn, input, "pytorch_rnn.onnx")