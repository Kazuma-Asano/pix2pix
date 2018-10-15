#coding:utf-8
from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
from   torch.autograd         import Variable
from   torch.utils.data       import DataLoader
import torchvision.transforms as transforms
from   torchvision.utils      import save_image, make_grid

from   dataloader             import get_test_set
from   networks               import define_G

import torch.backends.cudnn as cudnn
from termcolor import cprint


# Training settings
parser = argparse.ArgumentParser(description='a fork of pytorch pix2pix')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--checkpoint', '-c', type=str, default='checkpoint/netG_model_epoch_200.pth', help='model file to use')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
opt = parser.parse_args()
print(opt)

gpu_ids = []
# GPU enabled
cuda = torch.cuda.is_available()
if cuda:
    gpu_ids=[0]

cprint('==> Preparing Data Set', 'yellow')
root_path = './dataset/'
val_set = get_test_set(root_path)
val_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads,
                                    batch_size=opt.testBatchSize, shuffle=False)

cprint('==> Preparing Data Set: Complete\n', 'green')



print('==> Building model')

netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, norm='batch', use_dropout=False, gpu_ids=gpu_ids)
netG.load_state_dict(torch.load(opt.checkpoint))

if cuda:
    netG = netG.cuda()

print('------- OK ----------')


def eval():
    netG.eval()
    for i, batch in enumerate(val_data_loader, 1):

        with torch.no_grad():
            input = Variable(batch[0]).view(opt.testBatchSize, -1, 256, 256)
            gt = Variable(batch[1]).view(opt.testBatchSize, -1, 256, 256)
            # print(input.size())
            # exit()

            if cuda:
                input = input.cuda()
                gt = gt.cuda()

            prediction = netG(input)

            valDir = 'result/'
            os.makedirs(valDir, exist_ok=True)
            imgList = [ input[0], prediction[0], gt[0] ]
            grid_img = make_grid(imgList)
            save_image(grid_img, valDir + '{}.png'.format(i))

        print('[ {} / {} ]'.format(i, len(val_data_loader)+1) )


print('------ Start Program ------')
eval()
print('------ Finish ------')
