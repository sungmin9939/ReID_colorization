import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from torch import dropout, nn, Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from loss import OriTripletLoss
from transformers import ViTModel, ViTConfig
import torch.utils.model_zoo as model_zoo
from models.utils import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


##################################################################################
# Modality-Specific Encoder
##################################################################################


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # original padding is 1; original dilation is 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                      stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
        model.load_state_dict(
            remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    return model


class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, 'layer'+str(i),
                            getattr(model_v, 'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer'+str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.thermal, 'layer'+str(i),
                            getattr(model_t, 'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:  # meaning parameter sharing starts from stage 0
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.thermal, 'layer'+str(i))(x)
            return x



##################################################################################
# Residual Decoder
##################################################################################

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, w_lrelu=0.2):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
            '''
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
            '''
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(w_lrelu, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                              stride, dilation=dilation, bias=self.use_bias)
        
        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic', w_lrelu=0.2):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation,
                                    pad_type=pad_type, res_type=res_type, w_lrelu=w_lrelu)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic', w_lrelu=0.2):
        super(ResBlock, self).__init__()

        model = []
        if res_type == 'basic' or res_type == 'nonlocal':
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm,
                                  activation=activation, pad_type=pad_type, w_lrelu=w_lrelu)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm,
                                  activation='none', pad_type=pad_type, w_lrelu=w_lrelu)]
        elif res_type == 'slim':
            dim_half = dim//2
            model += [Conv2dBlock(dim, dim_half, 1, 1, 0, norm='in',
                                  activation=activation, pad_type=pad_type, w_lrelu=w_lrelu)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm,
                                  activation=activation, pad_type=pad_type, w_lrelu=w_lrelu)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm,
                                  activation=activation, pad_type=pad_type, w_lrelu=w_lrelu)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in',
                                  activation='none', pad_type=pad_type, w_lrelu=w_lrelu)]
        else:
            ('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResidualDecoder(nn.Module):
    def __init__(self, n_upsample=0, n_res=0, input_dim=1024, output_dim=3, dropout=0.1, res_norm='adain', activ='relu', pad_type='zero', res_type='basic', non_local=False, dec_type=1, init='kaiming', w_lrelu=0.2, mlp_input=0, mlp_output=0, mlp_dim=0, mlp_n_blk=0, mlp_norm='none', mlp_activ=''):
        super(ResidualDecoder, self).__init__()
        
        self.model = []       
        
        if dropout > 0:
            self.model += [nn.Dropout(p=dropout)]
        self.model += [ResBlocks(n_res, input_dim, res_norm, activ,
                                 pad_type=pad_type, res_type=res_type, w_lrelu=w_lrelu)]
        
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(input_dim, input_dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type, w_lrelu=w_lrelu)]
            input_dim //= 2
        
        # use reflection padding in the last conv layer
        if dec_type == 1:
            self.model += [Conv2dBlock(input_dim, output_dim, 7, 1, 3, norm='none',
                                       activation='tanh', pad_type=pad_type, w_lrelu=w_lrelu)]
        elif dec_type == 2:
            self.model += [Conv2dBlock(input_dim, input_dim, 3, 1, 1, norm='none',
                                       activation=activ, pad_type=pad_type, w_lrelu=w_lrelu)]
            self.model += [Conv2dBlock(input_dim, input_dim, 3, 1, 1, norm='none',
                                       activation=activ, pad_type=pad_type, w_lrelu=w_lrelu)]
            self.model += [Conv2dBlock(input_dim, output_dim, 1, 1, 0, norm='none',
                                       activation='none', pad_type=pad_type, w_lrelu=w_lrelu)]
                                
        #self.to_img_feature = nn.Sequential(*self.to_img_feature) 
        self.model = nn.Sequential(*self.model)

        self.model.apply(weights_init_kaiming)

    def forward(self, feat):
        
        return self.model(feat)
