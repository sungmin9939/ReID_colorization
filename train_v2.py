import cv2
import numpy as np
from torch._C import device
from torch.optim import optimizer
from torchvision import transforms
from torchvision.transforms.transforms import Compose
from data_loader import SYSUData, RegDBData, TestData, IdentitySampler
from models.transformers_MA import Trans_VIReID, Trans_VIReID_v2, Trans_Colorization
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
    loss_recon = AverageMeter()
    loss_crecon = AverageMeter()
    loss_tri = AverageMeter()
    loss_id = AverageMeter()
    loss_train = AverageMeter()


    #Image Transformation
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
   
    
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    invTrans = transforms.Compose([
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
    np_model = Trans_Colorization(opt).to(device)

    
    if os.path.exists(opt.checkpoint):
        np_model.load_state_dict(torch.load('./checkpoint/start_v2.pth'))
    
    model = nn.DataParallel(np_model, device_ids=[0, 1, 2, 3])

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, sampler=sampler)
    gall_loader = DataLoader(gallset, batch_size=opt.batch_size, shuffle=False)
    query_loader = DataLoader(queryset, batch_size=opt.batch_size, shuffle=False)

    
    criterion_tri = TripletLoss_WRT()
    criterion_id = nn.CrossEntropyLoss()
    criterion_recon = nn.L1Loss()
    criterion_cl = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
    
    pdist = nn.PairwiseDistance(2)

    if opt.optim == 'sgd':
        optimizer = optim.SGD(model.parameters())
    elif opt.optim == 'Adam' and opt.recon:
        optimizer = optim.AdamW([{'params': np_model.thermal.parameters(), 'lr':0.0001},
                                 {'params': np_model.thermal_decoder.parameters()}], lr=opt.lr, weight_decay=opt.decay)
    
        #optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.decay)
        
    cmc_list = []
    best_rank1 = 0
    if opt.recon:
        w_recon = opt.w_recon
    for i in range(opt.epochs):
        trainloader = tqdm(dataloader)
        np_model.train()
        model.train()
        
        if i == 100 or i == 300:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
        
        

        for idx, (rgb, ir, label) in enumerate(trainloader):
    
            rgb = Variable(rgb).to(device)
            ir = Variable(ir).to(device)
            label = Variable(label).to(device)
            

            feat_specific, feat_shared, feat_id, recon, c_recon = model(rgb, ir)
            
            tri_loss = criterion_tri(torch.cat((feat_shared[0][:,0], feat_shared[1][:,0]),dim=0), torch.cat((label, label)))
            id_loss = criterion_id(feat_id[0], label) + criterion_id(feat_id[1], label)            
            total_loss = tri_loss[0] + id_loss
            
            
            if opt.recon:
            
                recon_loss = w_recon * (criterion_recon(rgb, recon[0]) + criterion_recon(ir, recon[1]))
                cross_loss = w_recon * (criterion_recon(rgb, c_recon[0]) + criterion_recon(ir, c_recon[1]))
                
                total_loss += recon_loss
                total_loss +=  cross_loss
                
                '''
                if i % 5 == 0 and idx == 0:
                    for j in range(16):
                        writer.add_image('rgb_{}'.format(i),invTrans(rgb[j]),j)
                        writer.add_image('ir_{}'.format(i),invTrans(ir[j]),j)
                        writer.add_image('re_rgb_{}'.format(i),invTrans(out_re[0][j]),j)
                        writer.add_image('re_ir_{}'.format(i),invTrans(out_re[1][j]),j)
                        writer.add_image('cross_rgb_{}'.format(i),invTrans(out_cross[0][j]),j)
                        writer.add_image('cross_ir_{}'.format(i),invTrans(out_cross[1][j]),j)
                        writer.add_image('cycle_rgb_{}'.format(i),invTrans(out_hat[0][j]),j)
                        writer.add_image('cycle_ir_{}'.format(i),invTrans(out_hat[1][j]),j)
                '''
            else:
                recon_loss = 0
                cross_loss = 0
                
                
            
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            if opt.recon:                
                loss_recon.update(recon_loss.item(), rgb.size(0)*2)
                loss_crecon.update(cross_loss.item(), rgb.size(0)*2)
                
                
            loss_id.update(id_loss, rgb.size(0)*2)
            loss_tri.update(tri_loss[0], rgb.size(0)*2)
            loss_train.update(total_loss, rgb.size(0)*2)
            
            

        writer.add_scalar('train_loss', loss_train.avg, i)
        writer.add_scalar('tri_loss', loss_tri.avg, i)
        writer.add_scalar('id_loss', loss_id.avg, i)
        if opt.recon:
            writer.add_scalar('recon_loss', loss_recon.avg, i)
            writer.add_scalar('cross_loss',loss_crecon.avg, i)
        
        print(
            'epoch: {}\ntrain_loss: {}\nrecon_loss: {}\ncrecon_loss: {}\ntri_loss: {}\nid_loss: {}'.format(i, \
                loss_train.avg, loss_recon.avg, loss_crecon.avg, loss_tri.avg, loss_id.avg)
        )
            

        #test
        print("Testing model Accuracy...")
        np_model.eval()
        model.eval()
        ptr = 0
        gall_feat = np.zeros((len(gallset),768))
        gall_feat_att = np.zeros((len(gallset),206))
        gall_label = np.zeros(len(gallset))
        gall_names = []
        with torch.no_grad():
            for idx, (img, label, img_name) in enumerate(gall_loader):
                batch_num = img.size(0)
                img = Variable(img).to(device)
                feat, feat_att = np_model(img,img,modal=2)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                gall_label[ptr:ptr + batch_num] = label.numpy()
                ptr = ptr + batch_num
    
        ptr = 0
        query_feat = np.zeros((len(queryset), 768))
        query_feat_att = np.zeros((len(queryset), 206))
        query_label = np.zeros(len(queryset))
        query_names = []
        with torch.no_grad():
            for batch_idx, (input, label, img_name) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input).to(device)
                feat, feat_att = np_model(input, input, modal=1)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                query_label[ptr:ptr + batch_num] = label.numpy()
                ptr = ptr + batch_num

        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label, query_names, gall_names)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label, query_names, gall_names)

        writer.add_scalar('rank1', cmc[0], i)
        writer.add_scalar('mAP', mAP, i)
        writer.add_scalar('mINP', mINP, i)
        writer.add_scalar('rank1_att', cmc_att[0], i)
        writer.add_scalar('mAP_att', mAP_att, i)
        writer.add_scalar('mINP_att', mINP_att, i)
        
        if cmc[0] > best_rank1:
            best_rank1 = cmc[0]
            torch.save(np_model.state_dict(), "{}/best.pth".format(checkpoint_dir))
        torch.save(np_model.state_dict(), "{}/resume.pth".format(checkpoint_dir))
        
        cmc_list.append(cmc[0])
        
        
            
        print('rank1: {}'.format(cmc[0]))
        print('mAP: {}'.format(mAP))
        print('mINP: {}'.format(mINP))
        print('rank1_att: {}'.format(cmc_att[0]))
        print('mAP_att: {}'.format(mAP_att))
        print('mINP_att: {}'.format(mINP_att))
        
        ## draw sample basic
        if opt.recon:
            
            draw_sample_indices = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
        
            test_images_display_g = torch.stack(
                [gall_loader.dataset[draw_sample_indices[j]][0] for j in range(len(draw_sample_indices))]
            )
            test_images_display_q = torch.stack(
                [query_loader.dataset[draw_sample_indices[j]][0] for j in range(len(draw_sample_indices))]
            )
        
            num_images = test_images_display_g.size(0)
            
            gall_recon, query_recon, gall_crecon, query_crecon = [],[],[],[]
        
        
            with torch.no_grad():
                for j in range(num_images):
                    gall = test_images_display_g[j].unsqueeze(0).to(device)
                    query = test_images_display_q[j].unsqueeze(0).to(device)
                    
                    r,cr = np_model(query, gall, modal=0, draw_test=True)
                    
                    gall_recon.append(r[1].cpu())
                    query_recon.append(r[0].cpu())
                    gall_crecon.append(cr[1].cpu())
                    query_crecon.append(cr[0].cpu())
                
            
            gall_recon, query_recon = torch.cat(gall_recon), torch.cat(query_recon)
            gall_crecon, query_crecon = torch.cat(gall_crecon), torch.cat(query_crecon)
            
            exp = test_images_display_g, gall_recon, gall_crecon, test_images_display_q, query_recon, query_crecon
            
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
    parser.add_argument('--trial',default=9,type=int)

    parser.add_argument('--dim', default=768)
    parser.add_argument('--img_h', default=256, type=int)
    parser.add_argument('--img_w',default=128, type=int)
    parser.add_argument('--patch_size',default=16)
    parser.add_argument('--in_channel',default=3)
    parser.add_argument('--recon', default=True, type=bool)
    parser.add_argument('--batch_size',default=64, type=int)
    parser.add_argument('--margin',default=0.5)
    parser.add_argument('--version',default=2, type=int)
    ##loss weights
    parser.add_argument('--w_recon',default=50.0, type=float)
    

    opt = parser.parse_args()



    main(opt)