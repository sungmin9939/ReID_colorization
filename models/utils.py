import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1 and hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        '''
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
        '''
        
def make_pairs(x):
    """make the int -> tuple 
    """
    return x if isinstance(x, tuple) else (x, x)

def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
    for key, value in list(state_dict.items()):
        if key.startswith('fc.'):
            del state_dict[key]
    return state_dict