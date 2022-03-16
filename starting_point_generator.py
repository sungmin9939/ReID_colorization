from transformers import ViTModel
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchvision import transforms
import argparse
from models.transformers_MA import Trans_Colorization

parser = argparse.ArgumentParser()
parser.add_argument('--preconv',default='resnet', type=str)
parser.add_argument('--dropout',default=0.5, type=float)
parser.add_argument('--dataset',default='RegDB', type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--decay',default=0.0005, type=float)
parser.add_argument('--optim',default='Adam', type=str)
parser.add_argument('--checkpoint',default='./checkpoint/')
parser.add_argument('--epochs', default=70)
parser.add_argument('--log_path', default='./runs/')
parser.add_argument('--trial',default=6,type=int)

parser.add_argument('--dim', default=768)
parser.add_argument('--img_h', default=256, type=int)
parser.add_argument('--img_w',default=128, type=int)
parser.add_argument('--patch_size',default=16)
parser.add_argument('--in_channel',default=3)
parser.add_argument('--recon', default=True, type=bool)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--margin',default=0.5)
    

opt = parser.parse_args()


opt = parser.parse_args()
device = torch.device('cuda:0')


model = Trans_Colorization(opt)
torch.save(model.state_dict(), './checkpoint/start_color.pth')

input = torch.randn(2,3,256,128)
output = model(input)
print(output[1].shape)


'''
#model.load_state_dict(torch.load('./checkpoint/start.pth'))
#backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

x = torch.randn(2,3,256,128).to(device)
y = torch.randn(1,3,256,128).to(device)

feat, feat_att = model(x,y,modal=1)
print(feat.shape)
'''