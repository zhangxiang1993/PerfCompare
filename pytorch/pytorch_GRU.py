import torch
import torch.onnx

model = torch.nn.GRU(input_size=3,
                     hidden_size=16,
		             num_layers=1)
x = torch.randn(10, 1, 3)
h = torch.zeros(1, 1, 16)

print(model(x, h)) # produces no errors, prints outputs

torch.onnx.export(model, (x, h), 'temp.onnx', export_params=True, verbose=True)