#coding:utf-8
import os
from os import listdir
from os.path import join

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader # for test
from torch.autograd import Variable # for test
from torchvision.utils import save_image
from util import is_image_file, load_img

# for color print texts in console
from termcolor import cprint

"""
train.py もしくは test.py にて,

###
from dataloader import get_training_set, get_test_set
###
                                  #############
で import した後，このスクリプトの一番下の ## testing ## のように書くことで呼び出せる
                                  #############

入力画像 => GroundTruth のように学習する場合

## Input Image ##
"input_image"

## GroundTruth Image ##
"gt_img"

    root_path/  --- /train/ --- /input_img/
                 |           └- /gt_img/
                 |
                 └- /test/ --- /input_img/
                 |          └- /gt_img/
                 |
                 └- /val/ --- /input_img/

"""

class DatasetFromFolder(data.Dataset):
    def __init__(self, data_directory):
        super(DatasetFromFolder, self).__init__()
        self.input_img_path = join(data_directory, 'input_img')
        self.gt_img_path = join(data_directory, 'gt_img')

        self.input_img_filenames = [x for x in listdir(self.input_img_path) if is_image_file(x)]
        self.input_img_filenames.sort()
        self.gt_img_filenames = [x for x in listdir(self.gt_img_path) if is_image_file(x)]
        self.gt_img_filenames.sort()

        transform_list = [transforms.ToTensor(),]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.input_img_filenames)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.input_img_path, self.input_img_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.gt_img_path, self.gt_img_filenames[index]))
        target = self.transform(target)
        return input, target

class ValDatasetFromFolder(data.Dataset):
    def __init__(self, data_directory):
        super(ValDatasetFromFolder, self).__init__()
        self.input_img_path = join(data_directory, 'input_img')

        self.input_img_filenames = [x for x in listdir(self.input_img_path) if is_image_file(x)]
        self.input_img_filenames.sort()

        transform_list = [transforms.ToTensor(),
                          ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.input_img_filenames)

    def __getitem__(self, index):
        # Load Image
        input = load_img(join(self.input_img_path, self.input_img_filenames[index]))
        input = self.transform(input)
        return input


def get_training_set(root_dir):
    train_dir = join(root_dir, 'train')
    return DatasetFromFolder(train_dir)

def get_test_set(root_dir):
    test_dir = join(root_dir, 'test')
    return DatasetFromFolder(test_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, 'val')
    return ValDatasetFromFolder(val_dir)


if __name__ == '__main__':
    ###########
    # testing #
    ###########
    """
    make dataset folder
    root_path/  --- /train/ --- /input_img/
                 |           └- /gt_img/
                 |
                 └- /test/ --- /input_pcd/
                            └- /gt_img/
    """

    cprint('==> Preparing Data Set', 'yellow')
    root_path = './dataset/'
    train_set = get_training_set(root_path)
    test_set = get_test_set(root_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=1,
                                        batch_size=4, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=1,
                                        batch_size=4, shuffle=False)
    cprint('==> Preparing Data Set: Complete\n', 'green')

    #################################
    ######### visualization #########
    #################################

    for iteration, batch in enumerate(training_data_loader, 1):
        a, b = batch[0], batch[1]
        a = Variable(a)
        b = Variable(b)

        input_images = 'inputs.png'
        input_image = 'input.png'
        gt_images = 'gts.png'
        gt_image = 'gt.png'

        testDataloaderDir = 'test/DataLoader/'
        os.makedirs(testDataloaderDir, exist_ok=True)
        save_image(a.data, testDataloaderDir + '{}'.format(input_images))
        save_image(b.data, testDataloaderDir + '{}'.format(gt_images))
        save_image(a.data[1], testDataloaderDir + '{}'.format(input_image))
        save_image(b.data[1], testDataloaderDir + '{}'.format(gt_image))
        if(iteration == 1): break
