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

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True))

  def forward(self, x):
    return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool2d(2),
      DoubleConv(in_channels, out_channels))

  def forward(self, x):
    return self.maxpool_conv(x)

class Up(nn.Module):
  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    else:
      self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
    diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)

class UNet(nn.Module):
  def __init__(self, n_channels, n_classes, bilinear=True):
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.inc = DoubleConv(n_channels, 8)
    self.down1 = Down(8, 16)
    self.down2 = Down(16, 32)
    self.down3 = Down(32, 64)
    self.down4 = Down(64, 128)
    self.up1 = Up(128 + 64, 64, bilinear)
    self.up2 = Up(64 + 32, 32, bilinear)
    self.up3 = Up(32 + 16, 16, bilinear)
    self.up4 = Up(16 + 8, 12, bilinear)
    self.outc = OutConv(12, n_classes)

  def forward(self, x):
    x1 = self.inc(x) 
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits
  
#%%
def one_hot(tensor, num_classes):
  if num_classes == 1:
    true_1_hot = torch.eye(num_classes + 1)[tensor.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    true_1_hot_f = true_1_hot[:, 0:1, :, :]
    true_1_hot_s = true_1_hot[:, 1:2, :, :]
    true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
  else:
    true_1_hot = torch.eye(num_classes)[tensor.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
  return true_1_hot
  
def dice_loss(true, logits, eps=1e-7):
  """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
  """
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

def dice_coeff(true, logits, eps=1e-7):
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
  dice_coeff = (2. * intersection / (cardinality + eps)).mean()
  return dice_coeff

def jaccard_loss(true, logits, eps=1e-7):
  """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
  """
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

def jaccard_coeff(true, logits, eps=1e-7):
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
  jacc_coeff = (intersection / (union + eps)).mean()
  return jacc_coeff

def tversky_loss(true, logits, alpha, beta, eps=1e-7):
  """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
  """
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

def tversky_coeff(true, logits, alpha, beta, eps=1e-7):
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
  tversky_coeff = (num / (denom + eps)).mean()
  return tversky_coeff
#%%
def train(model, device, dataloader, optimizer, epochs, batch_size, loss, PathSaveCheckpointer, sheduler):
  history = np.zeros((epochs, len(dataloader)), dtype = np.float32)
  n = len(dataloader)
  model.train()
    
  for epoch in range(epochs):    
    for i, (images_, masks_) in enumerate(dataloader):
      images_ = images_.type(torch.float32)/255.
      masks_ = masks_.type(torch.float32)

      images_ = images_.to(device)
      masks_ = masks_.to(device)

      optimizer.zero_grad()
      predict = model(images_)

      loss = loss(predict, masks_)

      history[epoch, i] = loss.item()

      loss.backward()
      optimizer.step()
        
      if i%100 == 0:
        logg.info('%d / %d, %d / %d, %f'%(epoch + 1, epochs, i + 1, n, history[epoch, i]))
      
    modelCheckpointer(PathSaveCheckpointer%(1 + epoch), model, optimizer, epoch = epoch, history = history[epoch])
    #sheduler.step()
    
  return history

#%%
def computationTotalParameters(model):
  """
  Вычисляем общее количество параметров в нейросети
  """
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  return pytorch_total_params

from torchsummary import summary

def print_summaryPTH(model, input_shapeLIST, device):
  summary(model, input_shapeLIST, batch_size=-1, device=device)
  return None

#%%
RootPath = '/home/vsevolod/Desktop/Dymas/Satellite'
PathData = fs.join(RootPath, 'NN/Data')
PathLogger = fs.join(RootPath, 'Training/Loggers')
PathCheckPointer = 'Training/Chechpointer'
NameLogger = 'TestStart.log'

batch_size = 212
num_workers = 8

train_data = readHDF5file(PathData, 'PartTrainData.hdf5', ['Images', 'Masks'])
test_data = readHDF5file(PathData, 'PartTestData.hdf5', ['Images', 'Masks'])
train_images, train_mask = train_data[0], train_data[1]
test_images, test_mask = test_data[0], test_data[1]

train_images = np.swapaxes(train_images, 1, -1)
train_mask = np.swapaxes(train_mask, 1, -1)
test_images = np.swapaxes(test_images, 1, -1)
test_mask = np.swapaxes(test_mask, 1, -1)


params_loader_train = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': num_workers}
params_loader_test = {'batch_size': 2*batch_size,
          'shuffle': False,
          'num_workers': num_workers}


train_data = ClassDataloader(train_images, train_mask)
test_data = ClassDataloader(test_images, test_mask)

train_loader = data.DataLoader(train_data, **params_loader_train)
test_loader = data.DataLoader(test_data, **params_loader_test)

PathSaveCheckpointer = fs.join(RootPath, PathCheckPointer, 'TestStart%d.pth')
epochs = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3, 6, bilinear = True)
computationTotalParameters(model)
print_summaryPTH(model, (3, 250, 250), 'cpu')

A = torch.empty((5, 3, 250, 250), dtype = torch.float32)
pred = model(A)

masks = torch.ones((5, 250, 250), dtype = torch.long)

weight = torch.empty((6,), dtype = torch.float32)
weight[0], weight[1], weight[2] = 0., 1./0.245, 1./0.732
weight[3], weight[4], weight[5] = 1./0.0095, 1./0.003, 1./0.0013

loss = torch.nn.CrossEntropyLoss(weight)

print(loss(pred, masks))

B = one_hot(masks, 6)



model = model.to(device)

loss = nn.CrossEntropyLoss()
#optimizer = opt.SGD(model.parameters(), lr = 0.05)
optimizer = opt.Adam(model.parameters())
sheduler = opt.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.2)

history = train(model, device, train_loader, optimizer, epochs, batch_size, \
                loss, PathSaveCheckpointer, sheduler)
  