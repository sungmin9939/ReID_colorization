from models.utils import *
import torch.nn.functional as F
import torch
from PIL import Image
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from transformers import ViTModel, ViTConfig
from models.resnet import visible_module, thermal_module, ResidualDecoder
from torch import dropout, nn, Tensor
from torchvision.transforms import Compose, Resize, ToTensor
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, num_bottleneck=512, w_lrelu=0.2):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(w_lrelu, inplace=True)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = nn.Linear(num_bottleneck, class_num)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        return self.classifier(x)


class ConvGeluBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
        """build the conv3x3 + gelu + bn module
        """
        super(ConvGeluBN, self).__init__()
        self.kernel_size = make_pairs(kernel_size)
        self.stride_size = make_pairs(stride_size)
        self.padding_size = make_pairs(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.kernel_size,
                      stride=self.stride_size,
                      padding=self.padding_size),
            nn.GELU(),
            nn.BatchNorm2d(self.out_channel)
        )
        self.model.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.model(x)
        return x


class CMTStem(nn.Module):
    """make the model conv stem module
    """

    def __init__(self, kernel_size, in_channel, out_channel, layers_num):
        super(CMTStem, self).__init__()
        self.model = []

        self.layers_num = layers_num
        self.model += [ConvGeluBN(
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            stride_size=make_pairs(2)
        )]
        self.model += [ConvGeluBN(kernel_size=kernel_size, in_channel=out_channel, out_channel=out_channel, stride_size=1)
                       for _ in range(self.layers_num)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Trans_Colorization(nn.Module):
    #resnet encoder for source image(thermal image)
    #resnet decoder for generating an colored image

    def __init__(self, opt):
        super().__init__()
        if opt.dataset == "RegDB":
            self.num_classes = 206
        elif opt.dataset == "SYSU":
            self.num_classes = 1
        self.img_h = opt.img_h
        self.img_w = opt.img_w
        self.img_c = opt.in_channel


        self.dn_scale = 16
        self.n_downsample = 4
        self.dim = 1024
        
        self.thermal_encoder = thermal_module(share_net=4) # output shape: (N, 1024, 16, 8)
        self.thermal_decoder = ResidualDecoder(n_upsample=self.n_downsample, n_res=4, input_dim=self.dim,
                                                output_dim=self.img_c, dropout=opt.dropout, res_norm='bn'
                                               ) # output shape: (N, 3, 256, 128)
    def forward(self, x):
        feat = self.thermal_encoder(x)
        colored = self.thermal_decoder(feat)
        
        return feat, colored
        
##############################################################
# Cascaded Refinement Network
##############################################################
def get_output_chn(super_resolution):
    chn = 1024 if super_resolution < 128 else 512

    if super_resolution == 512:
        chn = 128
    if super_resolution == 1024:
        chn = 32

    return chn
        
        
class CRN(nn.Module):
    
    def __init__(self, super_resolution = 128, groups = 6):
        '''

        :param super_resolution: the height(short edge) of output image
        '''
        super(CRN, self).__init__()

        self.super_resolution = super_resolution
        last_inp_chn = get_output_chn(super_resolution)

        resolutions = [4, 8, 16, 32, 64, 128]
        return_labels = [True, True, True, True, True]

        if super_resolution >= 512:
            return_labels.append(False)
        if super_resolution == 1024:
            return_labels.append(False)
        return_labels = return_labels[::-1]

        self.refine_block0 = refine_block(resolutions[0])
        self.refine_block1 = refine_block(resolutions[1])
        self.refine_block2 = refine_block(resolutions[2])
        self.refine_block3 = refine_block(resolutions[3])
        self.refine_block4 = refine_block(resolutions[4])
        self.refine_block5 = refine_block(resolutions[5])

        '''
        if super_resolution > 256:
            self.refine_block7 = refine_block(resolutions[7])

        if super_resolution > 512:
            self.refine_block8 = refine_block(resolutions[8])
        '''

        self.last_conv = nn.Conv2d(last_inp_chn, 3, kernel_size=1)

        self.kaiming_init_params()

        print(self)
    def forward(self, label):
        x = self.refine_block0(label)
        
        x = self.refine_block1(label, x)
        
        x = self.refine_block2(label, x)
        
        x = self.refine_block3(label, x)
        
        x = self.refine_block4(label, x)
        
        x = self.refine_block5(label, x)
        

        '''
        if self.super_resolution > 256:
            x = self.refine_block7(label, x)

        if self.super_resolution > 512:
            x = self.refine_block8(label, x)
        '''

        x = self.last_conv(x)
        
        x = (x + 1.0) / 2.0 * 255.0

        a = x.split(3, dim = 1)

        x = torch.cat(a, dim = 0)

        return x

    def kaiming_init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, 0.2, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class refine_block(nn.Module):

    def __init__(self, super_resolution):
        '''

        :param super_resolution: the height(short edge) of output feature_map
        '''
        super(refine_block, self).__init__()
        self.super_resolution = super_resolution
        x_grid = torch.linspace(-1, 1, self.super_resolution).repeat(2 * self.super_resolution, 1)
        y_grid = torch.linspace(-1, 1, 2 * self.super_resolution).view(-1, 1).repeat(
            1, self.super_resolution)
        self.grid = torch.cat((x_grid.unsqueeze(2), y_grid.unsqueeze(2)), 2)
        self.grid = self.grid.unsqueeze_(0)


        out_chn = get_output_chn(super_resolution)
        in_chn = 3 if super_resolution == 4 else get_output_chn(super_resolution // 2) + 3


        self.conv1 = nn.Conv2d(in_chn, out_chn, kernel_size = 3, stride = 1, padding = 1)
        self.ln1 = nn.LayerNorm(normalized_shape=[out_chn, super_resolution * 2, super_resolution])

        self.conv2 = nn.Conv2d(out_chn, out_chn, kernel_size=3, stride = 1, padding=1)
        self.ln2 = nn.LayerNorm(normalized_shape=[out_chn, super_resolution * 2, super_resolution])

    def forward(self, label, x = None):

        # perform bilinear interpolation to downsample
        #x_grid = torch.linspace(-1, 1, 2 * self.super_resolution).repeat(self.super_resolution, 1)
        #y_grid = torch.linspace(-1, 1, self.super_resolution).view(-1, 1).repeat(
         #   1, self.super_resolution * 2)
        #grid = torch.cat((x_grid.unsqueeze(2), y_grid.unsqueeze(2)), 2)
       # grid = self.grid.repeat(label.size()[0], 1, 1, 1)

        #print('interpolation details: grid:{}, label:{}'.format(grid.size(), label.size()))
        #print('type: grid:{}, label:{}'.format(grid.type(), label.type()))
        grid = self.grid.repeat(label.size(0), 1, 1, 1).cuda()
        #print('Label size {}'.format(label.size()))
        label_downsampled = F.grid_sample(label, grid)

        if self.super_resolution != 4:
            x = F.upsample(x, size=(self.super_resolution * 2, self.super_resolution),
                           mode = 'bilinear', align_corners = True)
            x = torch.cat((x, label_downsampled), 1)
        else:
            
            x = label_downsampled

        #print('Label size {}'.format(x.size()))
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x, 0.2)

        del label_downsampled
        del grid
        return x



class Trans_VIReID_v2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt.dataset == "RegDB":
            self.num_classes = 206
        elif opt.dataset == "SYSU":
            self.num_classes = 1
        self.img_h = opt.img_h
        self.img_w = opt.img_w
        self.patch_size = opt.patch_size
        self.patch_overlap = int(self.patch_size/2)
        self.original_ch = opt.in_channel
        if opt.preconv == "CMTStem":
            self.dn_scale = 2
            self.n_downsample = 1
            self.visible = CMTStem(3, 3, 32, 2)
            self.thermal = CMTStem(3, 3, 32, 2)
            self.inter_dim = 32
        elif opt.preconv == "resnet":
            self.dn_scale = 16
            self.n_downsample = 4
            self.visible = visible_module(share_net=4)
            self.thermal = thermal_module(share_net=4)
            self.inter_dim = 256
        '''
        self.scaled_h = int(
            (self.img_h/self.dn_scale-self.patch_overlap)/self.patch_overlap)
        self.sclaed_w = int(
            (self.img_w/self.dn_scale-self.patch_overlap)/self.patch_overlap)
        '''
        self.scaled_h = int(self.img_h / self.dn_scale)
        self.scaled_w = int(self.img_w / self.dn_scale)
        self.dim = opt.dim

        self.visible_1x1 = nn.Conv2d(1024, self.dim, 1, 1)
        self.thermal_1x1 = nn.Conv2d(1024, self.dim, 1, 1)

        self.shared_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')

        self.visible_decoder = ResidualDecoder(n_upsample=self.n_downsample, n_res=4, input_dim=self.dim,
                                               inter_dim=self.inter_dim, output_dim=self.original_ch,
                                               dropout=opt.dropout, input_h=self.scaled_h, input_w=self.scaled_w, res_norm='bn'
                                               )
        self.thermal_decoder = ResidualDecoder(n_upsample=self.n_downsample, n_res=4, input_dim=self.dim,
                                               inter_dim=self.inter_dim, output_dim=self.original_ch,
                                               dropout=opt.dropout, input_h=self.scaled_h, input_w=self.scaled_w, res_norm='bn'
                                               )

        self.batchnorm = nn.BatchNorm1d(self.scaled_w * self.scaled_h+1)
        #self.classifier = nn.Linear(self.dim, self.num_classes)
        self.classifier = ClassBlock(self.dim, self.num_classes)
        self.modality_knowledge = nn.Linear(1, self.dim)

        self.visible_1x1.apply(weights_init_kaiming)
        self.thermal_1x1.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.batchnorm.apply(weights_init_kaiming)
        self.modality_knowledge.apply(weights_init_classifier)

    def forward(self, x_rgb, x_ir, modal=0, draw_test=False):
        if modal == 0:

            rgb_specific = self.visible_1x1(
                self.visible(x_rgb)).flatten(2).transpose(1, 2)
            ir_specific = self.thermal_1x1(
                self.thermal(x_ir)).flatten(2).transpose(1, 2)
            
            
            shared_rgb = self.shared_encoder(rgb_specific, modal=1, interpolate_pos_encoding=True, pass_embedding=True).last_hidden_state
            shared_ir = self.shared_encoder(ir_specific, modal=2, interpolate_pos_encoding=True, pass_embedding=True).last_hidden_state
            
            rgb_recon = self.visible_decoder(rgb_specific, shared_rgb[:,1:])
            thermal_recon = self.thermal_decoder(ir_specific, shared_ir[:,1:])
            
            rgb_cross_recon = self.visible_decoder(rgb_specific, shared_ir[:,1:])
            thermal_cross_recon = self.thermal_decoder(ir_specific, shared_rgb[:,1:])
            
            if draw_test:
                return (rgb_recon, thermal_recon), (rgb_cross_recon, thermal_cross_recon)
            
            
            rgb_id = self.classifier(shared_rgb[:,0])
            ir_id = self.classifier(shared_ir[:,0])
            
            
            return (rgb_specific, ir_specific),(shared_rgb, shared_ir), (rgb_id, ir_id), (rgb_recon, thermal_recon), (rgb_cross_recon, thermal_cross_recon)
            

            ###########################################################

        elif modal == 1: ## RGB                
            rgb_specific = self.visible_1x1(
                self.visible(x_rgb)).flatten(2).transpose(1, 2)
            shared_rgb = self.shared_encoder(rgb_specific, modal=1, interpolate_pos_encoding=True, pass_embedding=True).last_hidden_state
            
            
            feat = shared_rgb[:,0]
            feat_att = self.classifier(self.batchnorm(shared_rgb)[:,0])
            
            return feat, feat_att
        elif modal == 2: ## IR
            ir_specific = self.thermal_1x1(
                self.thermal(x_ir)).flatten(2).transpose(1, 2)
            shared_ir = self.shared_encoder(ir_specific, modal=2, interpolate_pos_encoding=True, pass_embedding=True).last_hidden_state
            
            
            feat = shared_ir[:, 0]
            feat_att = self.classifier(self.batchnorm(shared_ir)[:,0])

            return feat, feat_att
        
            
            


class Trans_VIReID(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt.dataset == "RegDB":
            self.num_classes = 206
        elif opt.dataset == "SYSU":
            self.num_classes = 1
        self.img_h = opt.img_h
        self.img_w = opt.img_w
        self.patch_size = opt.patch_size
        self.patch_overlap = int(self.patch_size/2)
        self.scaled_h = int((self.img_h-self.patch_overlap)/self.patch_overlap)
        self.sclaed_w = int((self.img_w-self.patch_overlap)/self.patch_overlap)

        self.in_channel = opt.in_channel
        self.dim = opt.dim
        '''
        vit_config = ViTConfig()
        self.disc_encoder = ViTModel(vit_config)
        self.excl_encoder = ViTModel(vit_config)
        '''
        self.disc_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')
        self.excl_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')

        # self.disc_encoder.embeddings.rgb_embeddings

        self.to_img = nn.Sequential(
            Rearrange('b (h w) c -> b c h w',
                      h=self.scaled_h, w=self.sclaed_w),
            nn.ConvTranspose2d(self.dim, self.in_channel, kernel_size=(
                self.patch_size, self.patch_size), stride=(self.patch_overlap, self.patch_overlap))
        )

        self.batchnorm = nn.BatchNorm1d(self.sclaed_w * self.scaled_h+1)
        #self.classifier = nn.Linear(self.dim, self.num_classes)
        self.classifier = ClassBlock(self.dim, self.num_classes)
        self.modality_knowledge = nn.Linear(1, self.dim)

        self.classifier.apply(weights_init_classifier)
        self.batchnorm.apply(weights_init_kaiming)
        self.to_img.apply(weights_init_kaiming)
        self.modality_knowledge.apply(weights_init_classifier)

    def forward(self, x_rgb, x_ir, modal=0):
        if modal == 0:

            disc_rgb = self.disc_encoder(
                pixel_values=x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            disc_ir = self.disc_encoder(
                pixel_values=x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state

            excl_rgb = self.excl_encoder(
                x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            excl_ir = self.excl_encoder(
                x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state

            feat_rgb = self.batchnorm(disc_rgb)
            feat_ir = self.batchnorm(disc_ir)

            rgb_id = self.classifier(feat_rgb[:, 0])
            ir_id = self.classifier(feat_ir[:, 0])

            # reconstruction part
            re_rgb = self.to_img(disc_rgb[:, 1:] + excl_rgb[:, 1:])
            re_ir = self.to_img(disc_ir[:, 1:] + excl_ir[:, 1:])

            dr_ei = self.to_img(disc_rgb[:, 1:] + excl_ir[:, 1:])
            di_er = self.to_img(disc_ir[:, 1:] + excl_rgb[:, 1:])

            disc_rgb_hat = self.disc_encoder(
                pixel_values=dr_ei, modal=2, interpolate_pos_encoding=True).last_hidden_state
            disc_ir_hat = self.disc_encoder(
                pixel_values=di_er, modal=1, interpolate_pos_encoding=True).last_hidden_state

            excl_rgb_hat = self.excl_encoder(
                di_er, modal=1, interpolate_pos_encoding=True).last_hidden_state
            excl_ir_hat = self.excl_encoder(
                dr_ei, modal=2, interpolate_pos_encoding=True).last_hidden_state

            hat_rgb = self.to_img(disc_rgb_hat[:, 1:] + excl_rgb_hat[:, 1:])
            hat_ir = self.to_img(disc_ir_hat[:, 1:] + excl_ir_hat[:, 1:])

            # modality embedding knowledge
            rgb_knowledge = self.modality_knowledge(
                self.disc_encoder.embeddings.rgb_embeddings)
            ir_knowledge = self.modality_knowledge(
                self.disc_encoder.embeddings.ir_embeddings)

            # make center vector for each identity
            rgb_feat_center = None
            ir_feat_center = None

            '''
            for chunk in list(torch.split(feat_rgb[:,0], 4)):  #feat_rgb[:,0] (batch_size, 768) chunk (batch_size/4, 768)
                if rgb_feat_center is None:
                    rgb_feat_center = torch.mean(chunk - rgb_knowledge, dim=0, keepdim=True)
                else:
                    rgb_feat_center = torch.cat((rgb_feat_center, torch.mean(chunk-rgb_knowledge, dim=0, keepdim=True)), dim=0)
                
                    
            for chunk in list(torch.split(feat_ir[:,0],4)):
                if ir_feat_center is None:
                    ir_feat_center = torch.mean(chunk - ir_knowledge, dim=0, keepdim=True)
                else:
                    ir_feat_center = torch.cat((ir_feat_center, torch.mean(chunk-ir_knowledge, dim=0, keepdim=True)), dim=0)
            
            #feat_center (batch_size/4, 768)
            '''

            rgb_knowledged_id = self.classifier(feat_rgb[:, 0] - rgb_knowledge)
            ir_knowledged_id = self.classifier(feat_ir[:, 0] - ir_knowledge)

            return (disc_rgb, disc_ir),\
                (excl_rgb, disc_ir),\
                (feat_rgb[:, 0]-rgb_knowledge, feat_ir[:, 0]-ir_knowledge),\
                (rgb_id, ir_id),\
                (re_rgb, re_ir),\
                (di_er, dr_ei),\
                (disc_rgb_hat, disc_ir_hat),\
                (excl_rgb_hat, excl_ir_hat),\
                (hat_rgb, hat_ir),\
                (rgb_feat_center, ir_feat_center),\
                (rgb_knowledged_id, ir_knowledged_id)\

            '''
            recon_loss = self.recon_loss(re_rgb, x_rgb) + self.recon_loss(re_ir, x_ir)
            cross_recon_loss = self.recon_loss(x_rgb, di_er) + self.recon_loss(x_ir, dr_ei)
            id_loss = self.id_loss(rgb_id, label) + self.id_loss(ir_id, label)

            return recon_loss, cross_recon_loss, id_loss, rgb_id, ir_id
            '''

        elif modal == 1:
            disc_rgb = self.disc_encoder(
                pixel_values=x_rgb, modal=1, interpolate_pos_encoding=True).last_hidden_state
            feat = disc_rgb[:, 0]
            feat_att = self.classifier(self.batchnorm(disc_rgb)[:, 0])

            return feat, feat_att
        elif modal == 2:
            disc_ir = self.disc_encoder(
                pixel_values=x_ir, modal=2, interpolate_pos_encoding=True).last_hidden_state
            feat = disc_ir[:, 0]
            feat_att = self.classifier(self.batchnorm(disc_ir)[:, 0])

            return feat, feat_att
