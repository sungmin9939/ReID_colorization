import numpy as np
from PIL import Image
import torch.utils.data as data
import os, random
from torchvision import transforms
from torch.utils.data.sampler import Sampler
import random
from itertools import chain





class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + '/train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + '/train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + '/train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + '/train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        


class RegDBData(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.train_folder_path = self.data_dir + '/train_py'
        self.transform = transform
        self.ids = sorted(os.listdir(self.train_folder_path))
        

        self.visible_id_pairs = {}
        self.visible_files = []
        for id in self.ids:
            if id[0] == 'T':
                continue
            else:
                images = os.listdir(os.path.join(self.train_folder_path, id))
                for image in images:
                    self.visible_id_pairs[image] = id[2:]
                    self.visible_files.append(image)
        

    def __getitem__(self, index):
        visible_image = self.visible_files[index]
        
        label = self.visible_id_pairs[visible_image]

        thermal_image = random.choice(list(os.listdir(self.train_folder_path+ '/T_{}'.format(label))))
        
        #print('{} {} {}'.format(label, visible_image, thermal_image))
        visible_image = Image.open(os.path.join(self.train_folder_path, 'V_'+label, visible_image))
        thermal_image = Image.open(os.path.join(self.train_folder_path, 'T_'+label, thermal_image))

        visible_image = self.transform(visible_image)
        thermal_image = self.transform(thermal_image)

        return visible_image, thermal_image, int(label)

    def __len__(self):
        return len(self.visible_files)


class TestData(data.Dataset):
    def __init__(self, data_dir, mode='gallery', transform=None):
        self.data_dir = data_dir + '/' + mode
        self.identities = sorted(os.listdir(self.data_dir))
        self.images = []
        self.labels = []
        self.transform = transform
        
        for id in self.identities:
            images = os.listdir(os.path.join(self.data_dir, id))
            for img in images:
                self.images.append(os.path.join(self.data_dir, id, img))
                self.labels.append(id)
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        label = self.labels[index]
        img = self.transform(img)
        img_name = self.images[index]

        return img, int(label), img_name

    def __len__(self):
        return len(self.images)

class TestData_dum(data.Dataset):
    def __init__(self, data_dir, mode='T', transform=None):
        self.data_dir = data_dir
        self.identities = []
        for i in os.listdir(self.data_dir):
            if i[0] == 'T':
                self.identities.append(i[2:])
            else:
                continue
        self.images = []
        self.labels = []
        self.transform = transform

        for id in self.identities:
            images = os.listdir(os.path.join(self.data_dir, '{}_{}'.format(mode,id)))
            for img in images:
                self.images.append(os.path.join(self.data_dir, '{}_{}'.format(mode,id),img))
                self.labels.append(id)
    def __getitem__(self,index):
        img = Image.open(self.images[index])
        label = self.labels[index]

        img = self.transform(img)

        return img, int(label)
    
    def __len__(self):
        return len(self.images)

def make_chuck(slicing_point):
    list = [i for i in range(slicing_point-9,slicing_point)]
    chunk = random.sample(list, 4)
    for i in chunk:
        list.remove(i)

    chunk2 = random.sample(list,4)
    for i in chunk2:
        list.remove(i)
    chunk3 = random.sample(chunk,2) + random.sample(chunk2, 1) + [list[0]]
    return np.array([chunk, chunk2, chunk3])




class IdentitySampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset_length = len(dataset)
        self.checklist = range(self.dataset_length)
        self.slicing_points = []
        self.batch_size = batch_size
        for i in range(9,self.dataset_length,9):
            self.slicing_points.append(i)
        
        assert self.batch_size % 4 == 0

        
    def __iter__(self):
        id_chunk = None

        for pt in self.slicing_points:
            chunks = make_chuck(pt)
            if id_chunk is None:
                id_chunk = chunks
            else:
                id_chunk = np.vstack((id_chunk,chunks))
        
        first_chunk = None
        second_chunk = None
        third_chunk = None

        for i in range(0,id_chunk.shape[0],3):
            first_chunk = np.vstack((first_chunk, id_chunk[i])) if first_chunk is not None else id_chunk[i]
            second_chunk = np.vstack((second_chunk, id_chunk[i+1])) if second_chunk is not None else id_chunk[i+1]
            third_chunk = np.vstack((third_chunk, id_chunk[i+2])) if third_chunk is not None else id_chunk[i+2]
        np.random.shuffle(first_chunk)
        np.random.shuffle(second_chunk)
        np.random.shuffle(third_chunk)

        final_chunk = np.vstack((first_chunk, second_chunk, third_chunk))

        return iter(final_chunk.flatten().tolist())

    def __len__(self):
        return len(self.slicing_points) * 12