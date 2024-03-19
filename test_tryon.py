from __future__ import print_function

import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import cv2
from tqdm import tqdm
from random import random
import random
from PIL import Image
from PIL import ImageDraw
import pycocotools.mask as maskUtils
import math
import json
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.parallel
from torchvision import models
import functools
from torch.nn.utils import spectral_norm
import argparse
from torch import FloatTensor
from cv2 import sort
from importlib_metadata import files


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    # if normalize:
    #    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # else:
    #    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = (image_numpy + 1) / 2.0
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]

    return image_numpy


# Converts a one-hot tensor into a colorful label map


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    # label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230,
                                                                           150, 140), (70, 70, 70), (102, 102, 156),
                         (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153,
                                                                            153, 153), (153, 153, 153), (250, 170, 30),
                         (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220,
                                                                           20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def save_html(path_result):
    result_dir = path_result
    filenames = sorted(os.listdir(result_dir))
    fs = []
    for filename in filenames:
        # fs.append(os.path.join(result_dir.split('/')[-1], filename))
        fs.append(filename)

    # first_dirs = os.listdir(result_dir)
    # for first_dir in first_dirs:
    #     if os.path.isdir(os.path.join(result_dir, first_dir)) == False:
    #         continue
    #     sub_dirs = os.listdir(os.path.join(result_dir, first_dir))
    #     for sub_dir in sub_dirs:
    #         filenames = os.listdir(os.path.join(
    #             result_dir, first_dir, sub_dir))
    #         for filename in filenames:
    #             fs.append(os.path.join(first_dir, sub_dir, filename))

    title = 'result'
    height = 'auto'
    width = 'auto'
    outpath = os.path.join(result_dir, 'vis.html')

    str_ = "<p>%s</p>" % title
    for i in range(len(fs)):
        path = fs[i]
        str_ += "<img src='" + path + "' alt='" + path
        str_ += "'  height=%s width=%s padding: 50px 10px style='border:100px '>" % (
            height, width)
        str_ += "<br>"

    with open(outpath, 'w') as f:
        f.write(str_)


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='flow',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_gpus', type=int, default=2, help='the number of gpus')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32],
                                 help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=14, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--nproc_per_node', type=int, default=1, help='nproc_per_node is the number of gpus')
        self.parser.add_argument('--master_port', type=int, default=7129, help='the master port number')
        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='/kaggle/input/vitonhdmody/VITON_traindata')
        self.parser.add_argument('--resize_or_crop', type=str, default='none',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--warproot', type=str, default='/kaggle/input/warping-results-mody-cloth-mask')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=4,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')
        self.parser.add_argument('--tv_weight', type=float, default=0.1, help='weight for TV loss')

        self.parser.add_argument('--image_pairs_txt', type=str,
                                 default='/kaggle/input/vitonhdmody/VITON_traindata/test_pairs_unpaired_1018.txt')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])  # to avoid any kernels' arguments
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # util.mkdirs(expr_dir)
        # if save and not self.opt.continue_train:
        #     file_name = os.path.join(expr_dir, 'opt.txt')
        #     with open(file_name, 'wt') as opt_file:
        #         opt_file.write('------------ Options -------------\n')
        #         for k, v in sorted(args.items()):
        #             opt_file.write('%s: %s\n' % (str(k), str(v)))
        #         opt_file.write('-------------- End ----------------\n')
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument(
            '--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
        self.parser.add_argument('--local_rank', type=int, default=0)

        self.parser.add_argument('--write_loss_frep', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=20,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument(
            '--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=50,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--lr', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument(
            '--lr_D', type=float, default=0.00005, help='initial learning rate for adam')
        self.parser.add_argument('--pretrain_checkpoint_D', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_warp_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PFAFN_gen_checkpoint', type=str,
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_warp_checkpoint', type=str,
                                 default='checkpoints/gp-vton_partflow_vitonhd_usepreservemask_lrarms_1027/PBAFN_warp_epoch_121.pth',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--PBAFN_gen_checkpoint', type=str,
                                 default='/kaggle/input/gp-vton-dataset/checkpoints/checkpoints/gp-vton_gen_vitonhd_wskin_wgan_lrarms_1029/PBAFN_gen_epoch_201.pth',
                                 help='load the pretrained model from the specified location')

        self.parser.add_argument('--CPM_checkpoint', type=str)
        self.parser.add_argument('--CPM_D_checkpoint', type=str)

        self.parser.add_argument('--write_loss_frep_eval', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq_eval', type=int, default=100,
                                 help='frequency of showing training results on screen')

        self.parser.add_argument('--add_mask_tvloss', action='store_true',
                                 help='if specified, use employ tv loss for the predicted composited mask')

        # for discriminators
        self.parser.add_argument(
            '--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument(
            '--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument(
            '--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument(
            '--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true',
                                 help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true',
                                 help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0,
                                 help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--debug_test', action='store_true')
        self.parser.add_argument(
            '--image_test_pairs_txt', type=str, default='')
        self.parser.add_argument(
            '--image_pairs_txt_eval', type=str, default='')
        self.parser.add_argument('--use_preserve_mask_refine', action='store_true',
                                 help='if specified, use preserve mask to refine to the warp clothes')

        self.parser.add_argument('--repeat_num', type=int, default=6)
        self.parser.add_argument('--loss_ce', type=float, default=1)
        self.parser.add_argument('--loss_gan', type=float, default=1)

        self.parser.add_argument('--debug_train', action='store_true')
        self.parser.add_argument('--test_flip', action='store_true')

        self.parser.add_argument(
            '--first_order_smooth_weight', type=float, default=0.01)
        self.parser.add_argument(
            '--squaretv_weight', type=float, default=1)

        self.parser.add_argument('--mask_epoch', type=int, default=-1)
        self.parser.add_argument('--no_dynamic_mask', action='store_true')

        self.parser.add_argument('--resolution', type=int, default=512)
        self.parser.add_argument('--dataset', type=str, default='vitonhd')

        self.isTrain = True


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                                innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                    norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                                norm_layer=norm_layer)

        self.model = unet_block
        self.old_lr = opt.lr
        self.old_lr_gmm = 0.1 * opt.lr

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


def load_checkpoint_parallel(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(opt.local_rank))
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    # flip = random.random() > 0.5
    flip = 0
    return {'crop_pos': (x, y), 'flip': flip}


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [512, 384]
        transform_list.append(transforms.Resize(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class AlignedDataset(BaseDataset):
    def initialize(self, opt, mode='train'):
        self.opt = opt
        self.root = opt.dataroot
        self.warproot = opt.warproot
        self.resolution = opt.resolution

        if self.resolution == 512:
            self.fine_height = 512
            self.fine_width = 384
            self.radius = 8
        else:
            self.fine_height = 1024
            self.fine_width = 768
            self.radius = 16

        pair_txt_path = opt.image_pairs_txt
        if mode == 'train' and 'train' in opt.image_pairs_txt:
            self.mode = 'train'
        else:
            self.mode = 'test'
        with open(pair_txt_path, 'r') as f:
            lines = f.readlines()

        self.P_paths = []
        self.C_paths = []
        self.C_types = []
        for line in lines:
            p_name, c_name, c_type = line.strip().split()
            P_path = os.path.join(self.root, self.mode, 'image', p_name)
            C_path = os.path.join(self.root, self.mode, 'cloth', c_name)
            if self.resolution == 1024:
                P_path = P_path.replace('.png', '.jpg')
                C_path = C_path.replace('.png', '.jpg')
            self.P_paths.append(P_path)
            self.C_paths.append(C_path)
            self.C_types.append(c_type)

        ratio_dict = None
        if self.mode == 'train':
            ratio_dict = {}
            person_clothes_ratio_txt = os.path.join(self.root, 'person_clothes_ratio_train.txt')
            with open(person_clothes_ratio_txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                c_name, ratio = line.strip().split()
                ratio = float(ratio)
                ratio_dict[c_name] = ratio
        self.ratio_dict = ratio_dict
        self.dataset_size = len(self.P_paths)

    ############### get palm mask ################
    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[..., np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b - d) / 4, b + (c - a) / 4
        x2, y2 = a - (b - d) / 4, b - (c - a) / 4

        x3, y3 = c + (b - d) / 4, d + (c - a) / 4
        x4, y4 = c - (b - d) / 4, d - (c - a) / 4

        kps = [x1, y1, x2, y2]

        v0_x, v0_y = c - a, d - b
        v1_x, v1_y = x3 - x1, y3 - y1
        v2_x, v2_y = x4 - x1, y4 - y1

        cos1 = (v0_x * v1_x + v0_y * v1_y) / \
               (math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v1_x * v1_x + v1_y * v1_y))
        cos2 = (v0_x * v2_x + v0_y * v2_y) / \
               (math.sqrt(v0_x * v0_x + v0_y * v0_y) * math.sqrt(v2_x * v2_x + v2_y * v2_y))

        if cos1 < cos2:
            kps.extend([x3, y3, x4, y4])
        else:
            kps.extend([x4, y4, x3, y3])

        kps = np.array(kps).reshape(1, -1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints, h, w):
        # shoulder, elbow, wrist
        s_x, s_y, s_c = hand_keypoints[0]
        e_x, e_y, e_c = hand_keypoints[1]
        w_x, w_y, w_c = hand_keypoints[2]

        up_mask = np.ones((h, w, 1), dtype=np.float32)
        bottom_mask = np.ones((h, w, 1), dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((50, 50), np.uint8)
            else:
                kernel = np.ones((100, 100), np.uint8)
            up_mask = cv2.dilate(up_mask, kernel, iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[..., np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            if self.resolution == 512:
                kernel = np.ones((30, 30), np.uint8)
            else:
                kernel = np.ones((60, 60), np.uint8)
            bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[..., np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask + hand_bottom_mask)
                             == 2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, parsing, keypoints):
        h, w = parsing.shape[0:2]

        left_hand_keypoints = keypoints[[5, 6, 7], :].copy()
        right_hand_keypoints = keypoints[[2, 3, 4], :].copy()

        left_hand_up_mask, left_hand_bottom_mask = self.get_hand_mask(
            left_hand_keypoints, h, w)
        right_hand_up_mask, right_hand_bottom_mask = self.get_hand_mask(
            right_hand_keypoints, h, w)

        # mask refined by parsing
        left_hand_mask = (parsing == 15).astype(np.float32)
        right_hand_mask = (parsing == 16).astype(np.float32)

        left_palm_mask = self.get_palm_mask(
            left_hand_mask, left_hand_up_mask, left_hand_bottom_mask)
        right_palm_mask = self.get_palm_mask(
            right_hand_mask, right_hand_up_mask, right_hand_bottom_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    ############### get palm mask ################

    def __getitem__(self, index):
        C_type = self.C_types[index]

        # person image
        P_path = self.P_paths[index]
        P = Image.open(P_path).convert('RGB')
        P_np = np.array(P)
        params = get_params(self.opt, P.size)
        transform_for_rgb = get_transform(self.opt, params)
        P_tensor = transform_for_rgb(P)

        # person 2d pose
        pose_path = P_path.replace('/image/', '/openpose_json/')[:-4] + '_keypoints.json'
        with open(pose_path, 'r') as f:
            datas = json.load(f)
        pose_data = np.array(datas['people'][0]['pose_keypoints_2d']).reshape(-1, 3)

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = transform_for_rgb(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        Pose_tensor = pose_map

        # person 3d pose
        densepose_path = P_path.replace('/image/', '/dense/')[:-4] + '.png'
        dense_mask = Image.open(densepose_path).convert('L')
        transform_for_mask = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        dense_mask_tensor = transform_for_mask(dense_mask) * 255.0
        dense_mask_tensor = dense_mask_tensor[0:1, ...]

        # person parsing
        parsing_path = P_path.replace('/image/', '/parse/')[:-4] + '.png'
        parsing = Image.open(parsing_path).convert('L')
        parsing_tensor = transform_for_mask(parsing) * 255.0

        parsing_np = (parsing_tensor.numpy().transpose(1, 2, 0)[..., 0:1]).astype(np.uint8)
        palm_mask_np = self.get_palm(parsing_np, pose_data)

        person_clothes_left_sleeve_mask_np = (parsing_np == 21).astype(int) + \
                                             (parsing_np == 24).astype(int)
        person_clothes_torso_mask_np = (parsing_np == 5).astype(int) + \
                                       (parsing_np == 6).astype(int)
        person_clothes_right_sleeve_mask_np = (parsing_np == 22).astype(int) + \
                                              (parsing_np == 25).astype(int)
        person_clothes_mask_np = person_clothes_left_sleeve_mask_np + \
                                 person_clothes_torso_mask_np + \
                                 person_clothes_right_sleeve_mask_np
        left_arm_mask_np = (parsing_np == 15).astype(int)
        right_arm_mask_np = (parsing_np == 16).astype(int)
        hand_mask_np = (parsing_np == 15).astype(int) + (parsing_np == 16).astype(int)
        neck_mask_np = (parsing_np == 11).astype(int)

        person_clothes_left_sleeve_mask_tensor = torch.tensor(
            person_clothes_left_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_torso_mask_tensor = torch.tensor(person_clothes_torso_mask_np.transpose(2, 0, 1)).float()
        person_clothes_right_sleeve_mask_tensor = torch.tensor(
            person_clothes_right_sleeve_mask_np.transpose(2, 0, 1)).float()
        person_clothes_mask_tensor = torch.tensor(person_clothes_mask_np.transpose(2, 0, 1)).float()
        left_arm_mask_tensor = torch.tensor(left_arm_mask_np.transpose(2, 0, 1)).float()
        right_arm_mask_tensor = torch.tensor(right_arm_mask_np.transpose(2, 0, 1)).float()
        neck_mask_tensor = torch.tensor(neck_mask_np.transpose(2, 0, 1)).float()

        seg_gt_tensor = person_clothes_left_sleeve_mask_tensor * 1 + person_clothes_torso_mask_tensor * 2 + \
                        person_clothes_right_sleeve_mask_tensor * 3 + left_arm_mask_tensor * 4 + \
                        right_arm_mask_tensor * 5 + neck_mask_tensor * 6
        background_mask_tensor = 1 - (person_clothes_left_sleeve_mask_tensor + person_clothes_torso_mask_tensor + \
                                      person_clothes_right_sleeve_mask_tensor + left_arm_mask_tensor + right_arm_mask_tensor + \
                                      neck_mask_tensor)
        seg_gt_onehot_tensor = torch.cat([background_mask_tensor, person_clothes_left_sleeve_mask_tensor, \
                                          person_clothes_torso_mask_tensor, person_clothes_right_sleeve_mask_tensor, \
                                          left_arm_mask_tensor, right_arm_mask_tensor, neck_mask_tensor], 0)

        ### preserve region mask
        if self.opt.no_dynamic_mask or self.ratio_dict is None:
            preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                  [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27,
                                                   28]])
            preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
        else:
            pc_ratio = self.ratio_dict[self.C_paths[index].split('/')[-1][:-4] + '.png']
            if pc_ratio < 0.9:
                preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                      [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27,
                                                       28]])
                preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
            elif pc_ratio < 0.95:
                if random() < 0.5:
                    preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                          [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26,
                                                           27, 28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
                else:
                    preserve_mask_for_loss_np = np.array(
                        [(parsing_np == index).astype(int) for index in [1, 2, 3, 4, 7, 12, 14, 23, 26, 27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
            else:
                if random() < 0.1:
                    preserve_mask_for_loss_np = np.array([(parsing_np == index).astype(int) for index in
                                                          [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26,
                                                           27, 28]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)
                else:
                    preserve_mask_for_loss_np = np.array(
                        [(parsing_np == index).astype(int) for index in [1, 2, 3, 4, 7, 12, 14, 23, 26, 27]])
                    preserve_mask_for_loss_np = np.sum(preserve_mask_for_loss_np, axis=0)

        preserve_mask_np = np.array([(parsing_np == index).astype(int) for index in
                                     [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 23, 26, 27, 28]])
        preserve_mask_np = np.sum(preserve_mask_np, axis=0)

        preserve_mask1_np = preserve_mask_for_loss_np + palm_mask_np
        preserve_mask2_np = preserve_mask_for_loss_np + hand_mask_np
        preserve_mask3_np = preserve_mask_np + palm_mask_np

        preserve_mask1_tensor = torch.tensor(preserve_mask1_np.transpose(2, 0, 1)).float()
        preserve_mask2_tensor = torch.tensor(preserve_mask2_np.transpose(2, 0, 1)).float()
        preserve_mask3_tensor = torch.tensor(preserve_mask3_np.transpose(2, 0, 1)).float()

        ### clothes
        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_for_rgb(C)

        CM_path = C_path.replace('/cloth/', '/cloth_mask/')[:-4] + '.png'
        CM = Image.open(CM_path).convert('L')
        CM_tensor = transform_for_mask(CM)

        cloth_parsing_path = C_path.replace('/cloth/', '/cloth_parse/')[:-4] + '.png'
        cloth_parsing = Image.open(cloth_parsing_path).convert('L')
        cloth_parsing_tensor = transform_for_mask(cloth_parsing) * 255.0
        cloth_parsing_tensor = cloth_parsing_tensor[0:1, ...]

        cloth_parsing_np = (cloth_parsing_tensor.numpy().transpose(1, 2, 0)).astype(int)
        flat_cloth_left_mask_np = (cloth_parsing_np == 21).astype(int)
        flat_cloth_middle_mask_np = (cloth_parsing_np == 5).astype(int) + \
                                    (cloth_parsing_np == 24).astype(int) + \
                                    (cloth_parsing_np == 13).astype(int)
        flat_cloth_right_mask_np = (cloth_parsing_np == 22).astype(int)
        flat_cloth_label_np = flat_cloth_left_mask_np * 1 + flat_cloth_middle_mask_np * 2 + flat_cloth_right_mask_np * 3
        flat_cloth_label_np = flat_cloth_label_np / 3

        flat_cloth_left_mask_tensor = torch.tensor(flat_cloth_left_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_middle_mask_tensor = torch.tensor(flat_cloth_middle_mask_np.transpose(2, 0, 1)).float()
        flat_cloth_right_mask_tensor = torch.tensor(flat_cloth_right_mask_np.transpose(2, 0, 1)).float()

        flat_cloth_label_tensor = torch.tensor(flat_cloth_label_np.transpose(2, 0, 1)).float()

        WC_tensor = None
        WE_tensor = None
        AMC_tensor = None
        ANL_tensor = None
        if self.warproot:
            ### skin color
            face_mask_np = (parsing_np == 14).astype(np.uint8)
            skin_mask_np = (face_mask_np + hand_mask_np + neck_mask_np).astype(np.uint8)
            skin = skin_mask_np * P_np
            skin_r = skin[..., 0].reshape((-1))
            skin_g = skin[..., 1].reshape((-1))
            skin_b = skin[..., 2].reshape((-1))
            skin_r_valid_index = np.where(skin_r > 0)[0]
            skin_g_valid_index = np.where(skin_g > 0)[0]
            skin_b_valid_index = np.where(skin_b > 0)[0]

            skin_r_median = np.median(skin_r[skin_r_valid_index])
            skin_g_median = np.median(skin_g[skin_g_valid_index])
            skin_b_median = np.median(skin_b[skin_b_valid_index])

            arms_r = np.ones_like(parsing_np[..., 0:1]) * skin_r_median
            arms_g = np.ones_like(parsing_np[..., 0:1]) * skin_g_median
            arms_b = np.ones_like(parsing_np[..., 0:1]) * skin_b_median
            arms_color = np.concatenate([arms_r, arms_g, arms_b], 2).transpose(2, 0, 1)
            AMC_tensor = torch.FloatTensor(arms_color)
            AMC_tensor = AMC_tensor / 127.5 - 1.0

            # warped clothes
            warped_name = C_type + '___' + P_path.split('/')[-1] + '___' + C_path.split('/')[-1][:-4] + '.png'
            warped_path = os.path.join(self.warproot, warped_name)
            warped_result = Image.open(warped_path).convert('RGB')
            warped_result_np = np.array(warped_result)

            if self.resolution == 512:
                w = 384
            else:
                w = 768
            warped_cloth_np = warped_result_np[:, -2 * w:-w, :]
            warped_parse_np = warped_result_np[:, -w:, :]

            warped_cloth = Image.fromarray(warped_cloth_np).convert('RGB')
            WC_tensor = transform_for_rgb(warped_cloth)

            warped_edge_np = (warped_parse_np == 1).astype(np.uint8) + \
                             (warped_parse_np == 2).astype(np.uint8) + \
                             (warped_parse_np == 3).astype(np.uint8)
            warped_edge = Image.fromarray(warped_edge_np).convert('L')
            WE_tensor = transform_for_mask(warped_edge) * 255.0
            WE_tensor = WE_tensor[0:1, ...]

            arms_neck_label = (warped_parse_np == 4).astype(np.uint8) * 1 + \
                              (warped_parse_np == 5).astype(np.uint8) * 2 + \
                              (warped_parse_np == 6).astype(np.uint8) * 3

            arms_neck_label = Image.fromarray(arms_neck_label).convert('L')
            ANL_tensor = transform_for_mask(arms_neck_label) * 255.0 / 3.0
            ANL_tensor = ANL_tensor[0:1, ...]

        input_dict = {
            'image': P_tensor, 'pose': Pose_tensor, 'densepose': dense_mask_tensor,
            'seg_gt': seg_gt_tensor, 'seg_gt_onehot': seg_gt_onehot_tensor,
            'person_clothes_mask': person_clothes_mask_tensor,
            'person_clothes_left_mask': person_clothes_left_sleeve_mask_tensor,
            'person_clothes_middle_mask': person_clothes_torso_mask_tensor,
            'person_clothes_right_mask': person_clothes_right_sleeve_mask_tensor,
            'preserve_mask': preserve_mask1_tensor, 'preserve_mask2': preserve_mask2_tensor,
            'preserve_mask3': preserve_mask3_tensor,
            'color': C_tensor, 'edge': CM_tensor,
            'flat_clothes_left_mask': flat_cloth_left_mask_tensor,
            'flat_clothes_middle_mask': flat_cloth_middle_mask_tensor,
            'flat_clothes_right_mask': flat_cloth_right_mask_tensor,
            'flat_clothes_label': flat_cloth_label_tensor,
            'c_type': C_type,
            'color_path': C_path,
            'img_path': P_path,
        }
        if WC_tensor is not None:
            input_dict['warped_cloth'] = WC_tensor
            input_dict['warped_edge'] = WE_tensor
            input_dict['arms_color'] = AMC_tensor
            input_dict['arms_neck_lable'] = ANL_tensor

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return len(self.P_paths) // (self.opt.batchSize * self.opt.num_gpus) * (
                    self.opt.batchSize * self.opt.num_gpus)
        else:
            return len(self.P_paths)

    def name(self):
        return 'AlignedDataset'


def CreateDataset(opt):
    if opt.dataset == 'vitonhd':
        dataset = AlignedDataset()
        dataset.initialize(opt, mode='test')
        # dataset.initialize(opt, mode='test', stage='gen')
    return dataset


from pylab import imread, subplot, imshow, show

import matplotlib.pyplot as plt


def main():
    torch.distributed.init_process_group('gloo', init_method='env://')
    device = torch.device(f'cuda:{opt.local_rank}')
    train_data = CreateDataset(opt)
    train_sampler = DistributedSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                              num_workers=4, pin_memory=True, sampler=train_sampler)

    gen_model = ResUnetGenerator(36, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
    # gen_model.train()
    gen_model.eval()
    gen_model.cuda()
    load_checkpoint_parallel(gen_model, opt.PBAFN_gen_checkpoint)

    # gen_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gen_model).to(device)
    if opt.isTrain and len(opt.gpu_ids):
        model_gen = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])
    else:
        model_gen = gen_model

    for data in tqdm(train_loader):
        real_image = data['image'].cuda()
        clothes = data['color'].cuda()
        preserve_mask = data['preserve_mask3'].cuda()
        preserve_region = real_image * preserve_mask
        warped_cloth = data['warped_cloth'].cuda()
        warped_prod_edge = data['warped_edge'].cuda()
        arms_color = data['arms_color'].cuda()
        arms_neck_label = data['arms_neck_lable'].cuda()
        pose = data['pose'].cuda()

        gen_inputs = torch.cat([preserve_region, warped_cloth, warped_prod_edge, arms_neck_label, arms_color, pose], 1)

        gen_outputs = model_gen(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_prod_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        k = p_tryon

        bz = pose.size(0)
        for bb in range(bz):
            combine = k[bb].squeeze()

            cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
            rgb = (cv_img * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cloth_id = data['color_path'][bb].split('/')[-1]
            person_id = data['img_path'][bb].split('/')[-1]
            c_type = data['c_type'][bb]
            save_path = 'sample/' + opt.name + '/' + person_id + '___' + cloth_id[:-4] + '.png'
            cv2.imwrite(save_path, bgr)


opt = TrainOptions().parse()
os.makedirs('sample/' + opt.name, exist_ok=True)
torch.cuda.set_device(opt.local_rank)

main()
