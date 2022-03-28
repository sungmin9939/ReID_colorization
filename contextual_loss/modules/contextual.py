import torch
import torch.nn as nn

from .vgg import VGG19
from .. import functional as F
from ..config import LOSS_TYPES


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.1,
                 loss_type: str = 'cosine',
                 use_vgg: bool = False,
                 vgg_layer: str = ['relu2_2', 'relu3_2', 'relu4_2']):

        super(ContextualLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES,\
            f'select a loss type from {LOSS_TYPES}.'

        self.band_width = band_width

        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y, domain='target'):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            if domain == 'source':
                
            # picking up vgg feature maps
                x = getattr(self.vgg_model(x), self.vgg_layer[-1])
                y = getattr(self.vgg_model(y), self.vgg_layer[-1])

                return F.contextual_loss(x, y, self.band_width)
            else:
                total_loss = 0
                for layer in self.vgg_layer:
                    x_l = getattr(self.vgg_model(x), layer)
                    y_l = getattr(self.vgg_model(y), layer)
                    total_loss += F.contextual_loss(x_l, y_l, self.band_width)
                return total_loss


class Perceptual(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 use_vgg: bool = False,
                 vgg_layer: str = ['relu5_2']):

        super(Perceptual, self).__init__()

        self.l1 = nn.L1Loss()
    
        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3,\
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            x = getattr(self.vgg_model(x), self.vgg_layer[0])
            y = getattr(self.vgg_model(y), self.vgg_layer[0])
            
            return self.l1(x,y)