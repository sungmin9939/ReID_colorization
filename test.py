import contextual_loss as cl
import torch
from PIL import Image
from torchvision import transforms
img = Image.open('/workspace/ReID_colorization/datasets/RegDB_01/gallery/000/T_female_back_t_01102_88.bmp')
img = transforms.ToTensor()(img)
print(img.shape)