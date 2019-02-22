import torch
import time
from torch import nn

class CleanBasicRNN(nn.Module):
  def __init__(self, batch_size, n_inputs, n_neurons):
    super(CleanBasicRNN, self).__init__()
    
    self.rnn = nn.RNNCell(n_inputs, n_neurons)
    self.hx = torch.randn(batch_size, n_neurons)  # initialize hidden state
  
  def forward(self, X):
    output = []
    
    # for each time step
    for i in range(2):
      self.hx = self.rnn(X[i], self.hx)
      output.append(self.hx)
    
    return output, self.hx


FIXED_BATCH_SIZE = 4  # our batch size is fixed for now
N_INPUT = 3
N_NEURONS = 5

X_batch = torch.tensor([[[0, 1, 2], [3, 4, 5],
                         [6, 7, 8], [9, 0, 1]],
                        [[9, 8, 7], [0, 0, 0],
                         [6, 5, 4], [3, 2, 1]]
                        ], dtype=torch.float)  # X0 and X1

dummy_input = torch.randn(2, 4, 3)
model = CleanBasicRNN(FIXED_BATCH_SIZE, N_INPUT, N_NEURONS)

t = time.time()
output_val, states_val = model(X_batch)
print('pytorch rnn 2: ', (time.time()-t)*1000)

torch.onnx.export(model, dummy_input, "pytorch_rnn_2.onnx")
print(output_val)  # contains all output for all timesteps
print(states_val)  # contains values for final state or final timestep, i.e., t=1