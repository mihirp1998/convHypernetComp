import time
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import ipdb 
st = ipdb.set_trace
from new_dataset import face_forensics_dataloader
import autoencoder
from tensorboardX import SummaryWriter
import sys 
import numpy as np


def log_image(name, tbwriter, data, iteration):
    image_to_log = data[0]
    image_to_log *= torch.tensor(np.array([0.229, 0.224, 0.225]).reshape(3,1,1)).cuda()
    image_to_log += torch.tensor(np.array([0.485, 0.456, 0.406]).reshape(3,1,1)).cuda()
    tbwriter.add_image(name, image_to_log, iteration)


# These can be loaded from argparser when code is merged.
exp_name = sys.argv[1]
data_path = "/projects/katefgroup/datasets/faceforensics/original_sequences/youtube/c23/images"
batch_size = 4
num_workers = 10
num_epochs = 100000000000000
log_freq = 10
val_freq = 50
lr = 0.005


autoencoder = autoencoder.UNet(n_classes=3, padding=True, up_mode='upsample').cuda()
print(autoencoder.parameters())
optimizer = optim.SGD(autoencoder.parameters(), lr=lr, momentum=0.9)
train_dataset = face_forensics_dataloader(data_path, 'train')
val_dataset = face_forensics_dataloader(data_path, 'val')
train_dataset_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataset_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

tbwriter = SummaryWriter('log_faceforensics/{}'.format(exp_name))
loss_criterion = nn.MSELoss()
iteration = 0
val_iteration = 0
for epoch in range(num_epochs):
    iteration_in_epoch = 0
    for batch_id, imgs in enumerate(train_dataset_loader):
        
        imgs = imgs.cuda()
        out = autoencoder(imgs)
        loss = loss_criterion(imgs, out)
        tbwriter.add_scalar('Train/Loss', loss, iteration)
        print('Epoch: %s. Iteration in epoch: %s. Total Iteration %s.  Loss %s' %(epoch, iteration_in_epoch, iteration, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % log_freq == 0:
            log_image("Train/Reconstructed_image", tbwriter, out, iteration)
            log_image("Train/Input_image", tbwriter, imgs, iteration)
        
        if iteration % val_freq == 0:
            val_iteration_in_epoch = 0
            for batch_id, imgs in enumerate(val_dataset_loader):
                imgs = imgs.cuda()
                out = autoencoder(imgs)
                loss = loss_criterion(imgs, out)

                tbwriter.add_scalar('Val/Loss', loss, iteration)
                print('Epoch: %s. Iteration in epoch: %s. Total Iteration %s.  Val Loss %s' %(epoch, iteration_in_epoch, iteration, loss))
                log_image("Val/Reconstructed_image", tbwriter, out, val_iteration)
                log_image("Val/Input_image", tbwriter, imgs, val_iteration)
                
                val_iteration_in_epoch += 1
                val_iteration += 1
                if val_iteration_in_epoch > 10:
                    break



        
        iteration += 1
        iteration_in_epoch += 1
        




        


