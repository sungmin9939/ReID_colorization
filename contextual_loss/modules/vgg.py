from collections import namedtuple

import torch.nn as nn
import torchvision.models.vgg as vgg


##2-2, 3-2, 4-2

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(9): #relu2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14): #relu3_2
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23): #relu4_2 
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 36): #relu5_2
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_2 = h
        h = self.slice4(h)
        h_relu5_2 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu2_2', 'relu3_2',
                           'relu4_2', 'relu5_2'])
        out = vgg_outputs(h_relu2_2, h_relu3_2,
                          h_relu4_2, h_relu5_2)

        return out
