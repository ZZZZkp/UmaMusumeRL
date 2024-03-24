import onnx
import torch

from actor_critic_pytorch import Policy

model = Policy()

x = torch.randn(1, 50)
action, status = model(x)
input_names = ['Game Parameters']
output_names = ['Action Prob', 'value']
torch.onnx.export(model, x, 'actor_critic.onnx', input_names=input_names, output_names=output_names)