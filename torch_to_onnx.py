'''
Description: In User Settings Edit
Author: Qianen
Date: 2021-04-15 13:46:47
LastEditTime: 2021-04-26 16:29:58
LastEditors: Qianen
'''
import torch
import sys
from pathlib import Path
from torchsummary import summary

input_size = (12, 1,300,300)
model_path = Path(sys.argv[1]).resolve()
out_path = model_path.with_suffix('.onnx')
model = torch.load(model_path, map_location=torch.device('cuda'))
summary(model, input_size[1:])
for name,param in model.named_parameters():
   print(name)
dummy_input = torch.randn(*input_size, device='cuda')
torch.onnx._export(model, dummy_input, out_path, verbose=True,opset_version=11)
