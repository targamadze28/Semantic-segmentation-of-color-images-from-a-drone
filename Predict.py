#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:48:20 2020

@author: vsevolod
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

import os.path as fs
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torchvision as tv

import torch
from torch.autograd import Function, Variable

from numba import njit, float64, float32, vectorize, int32, bool_, prange, cuda, uint8
import numba as nb
from PIL import Image, ImageOps
import cv2

import logging as logg

#%%
def saveHDF5file(PathToSave, SavedFileName, list_group_name, data):
  num_group = len(list_group_name)
  num_data = len(data)
  if num_group != num_data:
   raise RuntimeError('Список имен групп и длина списка данных не соответствуют!')
  
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'w')
  for i, group in enumerate(list_group_name):
    ff.create_dataset(group, data = data[i])
  ff.close()
  return None

def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data
#%%
def modelCheckpointer(PathNameToSave, model, opt = None, epoch = None, 
                      batch = None, loss = None, criterion = None, history = None, time_now = None):
  """
  Чекпоинтер, сохраняем информацию во время обучения.
    PathNameToSave - путь и имя файла
    model - модель
    opt - оптимизатор
    epoch - номер эпохи
    batch - номер батча
    loss - значение функции потерь
    criterion - сама функция потерь
    history - история обучения на предыдущих итерациях
    time_now - время сохранения
  """
  torch.save({
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'batch': batch,
            'criterion': criterion,
            'history': history,
            'time_save': time_now
            }, PathNameToSave)
  return None

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

#%%
class ClassDataloader(data.Dataset):
  """
  Кастомный даталоадер, содержит три поля
  1) images -- исходное изображение
  3) int_matrix
  """
  def __init__(self, images, masks):
    self.images = images
    self.masks = masks
    self.n_samples = images.shape[0]

  def __len__(self):
    return self.n_samples

  def __getitem__(self, index):
    return self.images[index], self.masks[index]

class UNET(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv1 = self.contract_block(in_channels, 32, 7, 3)
    self.conv2 = self.contract_block(32, 64, 3, 1)
    self.conv3 = self.contract_block(64, 128, 3, 1)
    self.axpadd1 = nn.ZeroPad2d((2, 1, 2, 1))
    self.axpadd2 = nn.ZeroPad2d((1, 0, 1, 0))

    self.upconv3 = self.expand_block(128, 64, 3, 1)
    self.upconv2 = self.expand_block(64*2, 32, 3, 1)
    self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)
    self.helpconv = nn.Conv2d(6, 6, kernel_size=(9, 9), stride=1, padding=1)


  def __call__(self, x):
        # downsampling part
    conv1 = self.conv1(x)
    conv1_ax = self.axpadd1(conv1)
    conv2 = self.conv2(conv1)
    conv2_ax = self.axpadd2(conv2)
    conv3 = self.conv3(conv2)

    upconv3 = self.upconv3(conv3)

    upconv2 = self.upconv2(torch.cat([upconv3, conv2_ax], 1))

    upconv1 = self.upconv1(torch.cat([upconv2, conv1_ax], 1))
    ext = self.helpconv(upconv1)    
    return ext

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
def dice_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1 - tversky_loss)

def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((size/2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size))  
    k = k * ( 1.0 / np.sum(k) )        
    return cv2.filter2D(image, -1, k)
#%%
def train(model, device, train_loader, test_loader, optimizer, epochs, batch_size, loss, PathSaveCheckpointer, sheduler):
  history = np.zeros((epochs, len(train_loader)), dtype = np.float32)
  n = len(train_loader)
  
  for epoch in range(epochs):    
    model.train()
    for i, (images_, masks_) in enumerate(train_loader):
      images_ = images_.type(torch.float32)/255.
      masks_ = masks_.type(torch.int64)

      images_ = images_.to(device)
      images_ += torch.empty_like(images_).normal_(0.1, 0.1)
      #images_ = torch.bernoulli(images_)
      images_ = images_/torch.max(images_)
      masks_ = masks_.to(device)

      optimizer.zero_grad()
      predict = model(images_)

      loss_tmp = dice_loss(masks_, predict)

      history[epoch, i] = loss_tmp.item()

      loss_tmp.backward()
      optimizer.step()
        
      if i%100 == 0:
        logg.info('%d / %d, %d / %d, %f'%(epoch + 1, epochs, i + 1, n, history[epoch, i]))
      
    modelCheckpointer(PathSaveCheckpointer%(1 + epoch), model, optimizer, epoch = epoch, history = history[epoch])
    sheduler.step()

    model.eval()
    n_test = len(test_loader)
    eval_predict = torch.zeros([n_test, 2], \
                          device='cpu')

    for j, (images_test, masks_test) in enumerate(test_loader):
      images_test = images_test.type(torch.float32)/255.
      masks_test = masks_test.type(torch.int64)

      images_test = images_test.to(device)
      masks_test = masks_test.to(device)
      with torch.no_grad():
        predict = model(images_test)
      eval_predict[j, 0] = 1 - dice_loss(masks_test, predict)
      eval_predict[j, 1] = 1 - jaccard_loss(masks_test, predict)
    logg.info('Test epoch %d dice: %f, jacard: %f'%(epoch + 1,\
                              torch.mean(eval_predict[:, 0]), torch.mean(eval_predict[:, 1])))
  return history

def evaluate(model, device, dataloader):
  model.eval()
  n_test = len(dataloader)
  eval_predict = torch.zeros([n_test, 2], \
                          device='cpu')

  for j, (images_test, masks_test) in enumerate(dataloader):
    images_test = images_test.type(torch.float32)/255.
    masks_test = masks_test.type(torch.int64)

    images_test = images_test.to(device)
    masks_test = masks_test.to(device)
    with torch.no_grad():
      predict = model(images_test)
    eval_predict[j, 0] = 1 - dice_loss(masks_test, predict)
    eval_predict[j, 1] = 1 - jaccard_loss(masks_test, predict)
    logg.info('Test dice: %f, jacard: %f'%(torch.mean(eval_predict[:, 0]),\
                                           torch.mean(eval_predict[:, 1])))
  return eval_predict

#%%
RootPath = '/data/vsevolod/DmiDiplom/SatelliteTask'
DataPath = fs.join(RootPath, 'Data')
DataFile = 'NewCropDataSet250x250.hdf5'
PathLogger = fs.join(RootPath, 'Training/Loggers')
PathCheckPointer = 'Training/Chechpointer'
NameLogger = 'UnetAdam.log'

batch_size = 128
num_workers = 4

logg.basicConfig(filename=fs.join(PathLogger, NameLogger), level=logg.INFO)
logg.info('Start script')
logg.info('Start loading dataset')

data_img = readHDF5file(DataPath, DataFile, ['Images', 'Masks'])
images, masks = data_img[0], data_img[1]

images = np.swapaxes(images, 1, -1)
masks = np.reshape(masks, (masks.shape[0], masks.shape[1], masks.shape[2]))

data_ind = readHDF5file(DataPath, 'SplitIndex.hdf5', ['train', 'test'])
train_ind, test_ind = data_ind[0], data_ind[1]

logg.info('Dataset was load, shapes: images %s, masks %s'%(str(images.shape), str(masks.shape)))

params_loader_train = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}

params_loader_test = {'batch_size': 2*batch_size,
          'shuffle': False,
          'num_workers': num_workers}


train_data = ClassDataloader(images[train_ind], masks[train_ind])
test_data = ClassDataloader(images[test_ind], masks[test_ind])

train_loader = data.DataLoader(train_data, **params_loader_train)
test_loader = data.DataLoader(test_data, **params_loader_test)

PathSaveCheckpointer = fs.join(RootPath, PathCheckPointer, 'UnetAdam%d.pth')
epochs = 35

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET(3, 6)
model = model.to(device)
PathToModel = '/data/vsevolod/DmiDiplom/SatelliteTask/Training/Chechpointer/UnetAdam35.pth'
checkpointer = Checkpointer(PathToModel, model)

evaluate = evaluate(model, device, test_loader)