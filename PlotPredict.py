#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:19:20 2020

@author: vsevolod
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from keras.utils import to_categorical

import os.path as fs
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

import torch
from torch.autograd import Function, Variable
#%%

def readUnionFile(PathToRead):
  ff = h5.File(fs.join(PathToRead, 'Random150samples.hdf5'), 'r')
  images = ff['Images'][...]
  masks = ff['Masks'][...]
  ff.close()
  return images, masks


def Checkpointer(PathNameFile, model, optimizer = None):
  """
    Загрузка чекпоинтера, точно загружается модель, если еще указать и оптимиза-
  тор, то загрузиться и он. 
    Два варианта возвращения: модель + чекпоинтер; модель + оптимизатор + чек-р. 
    Возвращаем чекпоинтер, чтобы можно из него доставать другие данные. 
    
    Чтобы что то достать из чекпоинтера, пишем:
            data = checkpoint['name data']
  """
  checkpointer = torch.load(PathNameFile)
  model.load_state_dict(checkpointer['model'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpointer['optimizer'])
    return model, optimizer, checkpointer
  
  return model, checkpointer


class UNET(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv1 = self.contract_block(in_channels, 32, 7, 3)
    self.conv2 = self.contract_block(32, 64, 3, 1)
    self.conv3 = self.contract_block(64, 128, 3, 1)

    self.upconv3 = self.expand_block(128, 64, 3, 1)
    self.upconv2 = self.expand_block(64*2, 32, 3, 1)
    self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

  def __call__(self, x):
        # downsampling part
    conv1 = self.conv1(x)
    conv2 = self.conv2(conv1)
    conv3 = self.conv3(conv2)

    upconv3 = self.upconv3(conv3)

    upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
    upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

    return upconv1

  def contract_block(self, in_channels, out_channels, kernel_size, padding):

    contract = nn.Sequential(
      torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
      torch.nn.BatchNorm2d(out_channels),
      torch.nn.ReLU(),
      torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
      torch.nn.BatchNorm2d(out_channels),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

    return contract

  def expand_block(self, in_channels, out_channels, kernel_size, padding):

    expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                           torch.nn.BatchNorm2d(out_channels),
                           torch.nn.ReLU(),
                           torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                           torch.nn.BatchNorm2d(out_channels),
                           torch.nn.ReLU(),
                           torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
    return expand
#%%
PathData = '/home/vsevolod/Desktop/Dymas/Satellite/NN/Data' 
PathToModel = '/home/vsevolod/Desktop/Dymas/Satellite/NN/Chechpointer/TestStartSGD19.pth'
images, masks = readUnionFile(PathData)

images = np.swapaxes(images, 1, -1)
#masks = np.reshape(masks, (masks.shape[0], masks.shape[1], masks.shape[2]))

device = torch.device('cpu')
model = UNET(3, 6)
model, checkpointer = Checkpointer(PathToModel, model)

print(checkpointer['history'].shape)

model = model.to(device)
model.eval()

images_ = torch.from_numpy(images)
masks_ = torch.from_numpy(masks)

images_ = images_.type(torch.float32)/255.
masks_ = masks_.type(torch.int64)

with torch.no_grad():
  pred = model(images_)

pred_np = pred.numpy()
pred_np = np.swapaxes(pred_np, -1, 1)


print(np.unique(masks))

masks_one_hot = to_categorical(masks)
print(np.unique(masks_one_hot[0, :, :, 5]))

plt.imshow(masks_one_hot[0, :, :, 1], cmap = plt.cm.gray)
plt.show()



i = 30


#masks_tmp = masks_one_hot[i]
masks_tmp = np.heaviside(pred_np[i] - 0.5, 1)

for k in range(6):
  masks_tmp[0, 0, k] = 0

#images = np.swapaxes(images, -1, 1)

fig, ax = plt.subplots(nrows=4, ncols=2)
fig.set_figwidth(15)    #  ширина и
fig.set_figheight(35)    #  высота "Figure"

fig.suptitle('Sample № %d'%(i), fontsize=10)
ax[0, 0].imshow(images[i])
ax[0, 0].set_title('source images')

ax[0, 1].imshow(255*masks_tmp[:, :, 0].astype(np.int32), cmap = plt.cm.gray)
ax[0, 1].set_title('class 0')

ax[1, 0].imshow(255*masks_tmp[:, :, 1].astype(np.int32), cmap = plt.cm.gray)
ax[1, 0].set_title('class 1')

ax[1, 1].imshow(255*masks_tmp[:, :, 2].astype(np.int32), cmap = plt.cm.gray)
ax[1, 1].set_title('class 2')

ax[2, 0].imshow(255*masks_tmp[:, :, 3].astype(np.int32), cmap = plt.cm.gray)
ax[2, 0].set_title('class 3')
  
ax[2, 1].imshow(255*masks_tmp[:, :, 4].astype(np.int32), cmap = plt.cm.gray)
ax[2, 1].set_title('class 4')

ax[3, 0].imshow(255*masks_tmp[:, :, 5].astype(np.int32), cmap = plt.cm.gray)
ax[3, 0].set_title('class 5')

plt.show()
