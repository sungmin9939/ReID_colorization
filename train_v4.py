from turtle import color
import cv2
import numpy as np
from torch._C import device
from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import SYSUData, RegDBData, TestData, IdentitySampler
from models.transformers_MA import Trans_Colorization, CRN
from utils import *
import torch.utils.data as data
import sys
import torchvision.utils as vutils
import argparse
import torch.optim as optim
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from loss import TripletLoss_WRT
from utils import AverageMeter, eval_regdb
from tensorboardX import SummaryWriter
import torch.nn as nn
import contextual_loss as cl


def main(opt):

    


    #Loss recoder
    loss_target = AverageMeter()
    loss_source = AverageMeter()
    loss_train = AverageMeter()


    #Image Transformation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        #transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
   
    
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    invTrans = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    ])


    device = torch.device('cuda:0')

    #Initiate dataset
    if opt.dataset == 'RegDB':
        data_path = './datasets/RegDB_01'
        dataset = RegDBData(data_path, transform_train)
        sampler = IdentitySampler(dataset, opt.batch_size)
    elif opt.dataset == 'sysu':
        data_path = './datasets/SYSU-MM01'
        dataset = SYSUData(data_path, transform_train)
    gallset = TestData('./datasets/RegDB_01','gallery', transform_test)
    queryset = TestData('./datasets/RegDB_01','query', transform_test)
    
    
    #Initiate loss logging directory
    suffix = opt.dataset
    suffix = suffix + '_trial{}_batch{}_epoch{}_version{}'.format(opt.trial, opt.batch_size, opt.epochs, opt.version)

    log_dir = opt.log_path + '/' + suffix + '/'
    train_samples = './train_result/' + suffix
    
    checkpoint_dir = './checkpoint/{}_trial{}_batch{}_epoch{}_v{}'.format(opt.dataset, opt.trial, opt.batch_size, opt.epochs, opt.version)
    
    
    if not os.path.isdir(train_samples):
        print("Making a Directory for Saving Training Samples: {}".format(train_samples))
        os.makedirs(train_samples)

    if not os.path.isdir(log_dir):
        print("Making a Directory for Logging: {}".format(log_dir))
        os.makedirs(log_dir)
    
    if not os.path.isdir(checkpoint_dir):
        print("Making a Directory for Saving Checkpoints: {}".format(checkpoint_dir))
        os.makedirs(checkpoint_dir)
    
    writer = SummaryWriter(log_dir)

    #make model
    np_model = CRN(opt.super_r).to(device)

    torch.save(np_model.state_dict(), './checkpoint/start_CRN.pth')
    np_model.load_state_dict(torch.load('./checkpoint/start_CRN.pth'))
    
    
    if opt.resume_epoch:
        print("load resume checkpoint from {}".format(checkpoint_dir))
        np_model.load_state_dict(torch.load(checkpoint_dir + '/resume.pth'))
    
    model = nn.DataParallel(np_model, device_ids=[0, 1, 2, 3])

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True) # no sampler
    gall_loader = DataLoader(gallset, batch_size=opt.batch_size, shuffle=False)
    query_loader = DataLoader(queryset, batch_size=opt.batch_size, shuffle=False)

    
    
    criterion_cl = cl.ContextualLoss(use_vgg=True, vgg_layer=['relu2_2', 'relu3_2', 'relu4_2']).to(device)
    criterion_per = cl.Perceptual(use_vgg=True).to(device)

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif opt.optim == 'Adam':
        optimizer = optim.AdamW([{'params': np_model.refine_block0.parameters()},
                                 {'params': np_model.refine_block1.parameters()},
                                 {'params': np_model.refine_block2.parameters()},
                                 {'params': np_model.refine_block3.parameters()},
                                 {'params': np_model.refine_block4.parameters()},
                                 {'params': np_model.refine_block5.parameters()},
                                 {'params': np_model.last_conv.parameters()}], lr=opt.lr, weight_decay=opt.decay)
    
        #optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.decay)
    
    start_epoch = opt.resume_epoch
    
    for i in range(start_epoch, opt.epochs + start_epoch):
        trainloader = tqdm(dataloader)
        np_model.train()
        model.train()
        
        '''
        if i == 100 or i == 300:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
                '''
        
        

        for idx, (rgb, ir, label) in enumerate(trainloader):
    
            rgb = Variable(rgb).to(device)
            ir = Variable(ir).to(device)
            label = Variable(label).to(device)
            

            colored = model(ir)
            
            #Semantic style transfer
            '''
            loss_to_target = criterion_cl(colored, rgb)
            loss_to_source = criterion_cl(colored, ir, domain='source')
            
            total_loss = loss_to_source + loss_to_target
            '''
            #Puppet control
            loss_to_target = criterion_cl(colored,rgb)
            loss_to_source = 0.1 * criterion_per(colored,rgb) #Actually perceptual loss
            
            total_loss = loss_to_source + loss_to_target
            optimizer.zero_grad()
            
            total_loss.backward()
            optimizer.step()


            
            loss_source.update(loss_to_source, rgb.size(0)*2)
            loss_target.update(loss_to_target, rgb.size(0)*2)
            loss_train.update(total_loss, rgb.size(0)*2)
            
        writer.add_scalar('train_loss', loss_train.avg, i)
        writer.add_scalar('to_target_loss', loss_target.avg, i)
        writer.add_scalar('to_source_loss', loss_source.avg, i)
        
        print(
            'epoch: {}\ntrain_loss: {}\nto_target_loss: {}\nto_source_loss: {}\n'.format(i, \
                loss_train.avg, loss_target.avg, loss_source.avg)
        )
        
        torch.save(np_model.state_dict(), "{}/resume.pth".format(checkpoint_dir))
            
        ## draw sample basic
        if i % 10 == 0:
                    
            draw_sample_indices = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
        
            test_images_display_g = torch.stack(
                [gall_loader.dataset[draw_sample_indices[j]][0] for j in range(len(draw_sample_indices))]
            )
            test_images_display_q = torch.stack(
                [query_loader.dataset[draw_sample_indices[j]][0] for j in range(len(draw_sample_indices))]
            )
        
            num_images = test_images_display_g.size(0)
            
            gall_colored = []
        
        
            with torch.no_grad():
                for j in range(num_images):
                    gall = test_images_display_g[j].unsqueeze(0).to(device)
                    
                    _, g_colored = np_model(gall)
                    
                    gall_colored.append(g_colored.cpu())
                
            gall_colored = torch.cat(gall_colored)
            
            exp = test_images_display_g, test_images_display_q, gall_colored
            
            image_tensor = torch.cat([images for images in exp])
            
            image_grid = vutils.make_grid(image_tensor.data, num_images, padding=0, normalize=True, scale_each=True)
            vutils.save_image(image_grid, '{}/sample_basic_{}epochs.png'.format(train_samples,i),1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preconv',default='resnet', type=str)
    parser.add_argument('--dropout',default=0.5, type=float)
    parser.add_argument('--dataset',default='RegDB', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay',default=0.0005, type=float)
    parser.add_argument('--optim',default='Adam', type=str)
    parser.add_argument('--checkpoint',default='./checkpoint/')
    parser.add_argument('--epochs', default=500)
    parser.add_argument('--log_path', default='./runs/')
    parser.add_argument('--trial',default=7,type=int)

    parser.add_argument('--dim', default=768)
    parser.add_argument('--img_h', default=256, type=int)
    parser.add_argument('--img_w',default=128, type=int)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--recon', default=True, type=bool)
    parser.add_argument('--batch_size',default=16, type=int)
    parser.add_argument('--margin',default=0.5)
    parser.add_argument('--version',default=3, type=int)
    ##loss weights
    parser.add_argument('--w_recon',default=50.0, type=float)
    parser.add_argument('--resume_epoch',default=0, type=int)
    parser.add_argument('--super_r',default=128, type=int)    

    opt = parser.parse_args()



    main(opt)