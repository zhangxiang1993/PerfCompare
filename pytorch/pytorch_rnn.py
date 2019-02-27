import torch
import time

def torch_rnn():
  RNN = torch.nn.RNN(150528, 32, 1)
  return RNN

def eval_torch_RNN():
  model = torch_rnn()
  input = torch.randn(1, 1, 150528)
  h0 = torch.randn(1, 1, 32)

  t = time.time()
  times_to_run = 1000
  for i in range(times_to_run):
    input = torch.randn(1, 1, 150528)
    h0 = torch.randn(1, 1, 32)
    output, hn = model(input, h0)
  print('pytorch rnn: ', (time.time()-t)*1000/times_to_run) # 140.5630111694336

  print (output.shape)
  torch.onnx.export(model, input, "pytorch_rnn.onnx")

