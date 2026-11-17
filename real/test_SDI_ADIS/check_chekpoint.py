import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_path', required=True)
opt = parser.parse_args()

checkpoint = torch.load(opt.pretrained_model_path)
print(checkpoint['args'])
