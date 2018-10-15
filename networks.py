#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda()
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids)  > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    netD = NLayerDiscriminator(input_nc, ndf, n_layer=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netD.cuda()
    netD.apply(weights_init)
    return netD

def print_network(net):
    num_parms = 0
    for parm in net.parameters():
        num_parms += parm.numel()
    print(net)
    print('Total number of parameters:{}'.format(num_parms))

################################################################################
################################## Generator ###################################
################################################################################

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [ nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                  norm_layer(ngf, affine=True),
                  nn.ReLU(inplace=True) ]

        ## DownSampling ##
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [ nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1),
                       norm_layer(ngf*mult*2, affine=True),
                       nn.ReLU(inplace=True) ]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ ResnetBlock(ngf*mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout) ]

        ## UpSampling ##
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [ nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf*mult/2), affine=True),
                       nn.ReLU(inplace=True) ]

        model += [ nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3) ]
        model += [ nn.Tanh() ]

        self.model = nn.Sequential(*model) # *list -> unpack ex) list -> [3], *list -> 3

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        assert(padding_type == 'zero')
        p = 1

        conv_block += [ nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim, affine=True),
                        nn.ReLU(inplace=True) ]

        if use_dropout:
            conv_block += [ nn.Dropout(0.5) ]

        conv_block += [ nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim, affine=True) ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

################################################################################
############################### Discriminator ##################################
################################################################################
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layer=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [ nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, inplace=True) ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layer):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            sequence += [ nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw),
                          norm_layer(ndf*nf_mult, affine=True),
                          nn.LeakyReLU(0.2, inplace=True) ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layer, 8)
        sequence += [ nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=1, padding=padw),
                      norm_layer(ndf*nf_mult, affine=True),
                      nn.LeakyReLU(0.2, inplace=True) ]

        sequence += [ nn.Conv2d(ndf*nf_mult, 1, kernel_size=kw, stride=1, padding=padw) ]

        if use_sigmoid:
            sequence += [ nn.Sigmoid() ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

########################################
if __name__ == '__main__':
    """
    testing
    """
    model = define_G(input_nc=3, output_nc=3, ngf=64, norm='batch', use_dropout=False, gpu_ids=[])
    # Input RGB(3ch) image
    x = Variable(torch.FloatTensor( np.random.random((1, 1, 256, 256)) ))
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
