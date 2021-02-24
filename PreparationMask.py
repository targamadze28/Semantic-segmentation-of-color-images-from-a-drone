#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:04:43 2020

@author: vsevolod
"""

import numpy as np
import h5py as h5
import os
import os.path as fs

import matplotlib.pyplot as plt

import cv2 as cv
from sklearn.preprocessing import OneHotEncoder

from keras.utils import to_categorical

def createListName(Path):
  list_name = os.listdir(Path)
  return np.sort(list_name).tolist()

def correctLsitname(lm):
  for i in range(len(lm)):
    lm[i] = lm[i][:-5] + '.JPG'
  return lm

def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data

def readAllMask(Path, list_name):
  list_mask = []
  for name in list_name:
    mask_tmp = readHDF5file(Path, name, ['Mask'])[0]
    list_mask.append(mask_tmp)
  return list_mask

def readAllIMG(Path, list_name):
  list_img = []
  for name in list_name:
    img_tmp = cv.imread(fs.join(Path, name))
    print(img_tmp.dtype)
    list_img.append(img_tmp)
  return list_img

def correctionShapes(images, num_chanell = 1):
  n = len(images)
  print(n)
  if num_chanell == 1:
    true_shape = (4000, 6000)
    images_new = np.empty((n, 4000, 6000), dtype = np.uint8)
  else:
    true_shape = (4000, 6000, 3)
    images_new = np.empty((n, 4000, 6000, 3), dtype = np.uint8)
  for i in range(n):
    print(images[i].shape)
    if images[i].shape != true_shape:
      img_tmp = images[i]
      images_new[i] = np.rot90(img_tmp)
    else:
      images_new[i] = images[i]
  return images_new

def saveMask(pathToSave, mask):
  ff = h5.File(fs.join(pathToSave, 'mask.hdf5'), 'w')
  ff.create_dataset('Mask', data = mask)
  ff.close()
  return None

def saveImage(pathToSave, image):
  ff = h5.File(fs.join(pathToSave, 'images.hdf5'), 'w')
  ff.create_dataset('images', data = image)
  ff.close()
  return None

def readMask(PathToRead):
  ff = h5.File(fs.join(PathToRead, 'mask.hdf5'), 'r')
  mask = ff['Mask'][...]
  ff.close()
  return mask

def readImages(PathToRead):
  ff = h5.File(fs.join(PathToRead, 'images.hdf5'), 'r')
  images = ff['images'][...]
  ff.close()
  return images

def saveUnionFile(PathToSave, images, mask):
  ff = h5.File(fs.join(PathToSave, 'DataSet.hdf5'), 'w')
  ff.create_dataset('Images', data = images)
  ff.create_dataset('Masks', data = mask)
  ff.close()
  return None

def readUnionFile(PathToRead):
  ff = h5.File(fs.join(PathToRead, 'DataSet.hdf5'), 'r')
  images = ff['Images'][...]
  masks = ff['Masks'][...]
  ff.close()
  return images, masks

def plotforcheck(PathToSave, images, masks):
  n = len(images)
  for i in range(n):
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(4, 2)
    fig.set_figwidth(15)    #  ширина и
    fig.set_figheight(35)    #  высота "Figure"

    fig.suptitle('Sample № %d'%(i), fontsize=10)

    ax1.imshow(images[i])
    ax1.set_title('source images')

    ax2.imshow(masks[i, :, :, 0])
    ax2.set_title('class 0')

    ax3.imshow(masks[i, :, :, 1])
    ax2.set_title('class 1')

    ax4.imshow(masks[i, :, :, 2])
    ax2.set_title('class 2')

    ax5.imshow(masks[i, :, :, 3])
    ax2.set_title('class 3')
  
    ax6.imshow(masks[i, :, :, 4])
    ax2.set_title('class 4')

    ax7.imshow(masks[i, :, :, 5])
    ax7.set_title('class 5')
    
    fig.savefig(fs.join(PathToSave, 'sample_%d.png'%(i))) 
  return None

#RootPathFile = '/data/vsevolod/DmiDiplom/SatelliteTask'
#RootPathImages = '/data/vsevolod/DmiDiplom/SatelliteTask/ExpDS'

RootPathFileMask = '/home/vsevolod/Desktop/Dymas/Satellite/Data/Mask'
RootPathImages = '/home/vsevolod/Desktop/Dymas/Satellite/Data/Images'


list_name = createListName(RootPathFileMask)
mask = readAllMask(RootPathFileMask, list_name)
mask = correctionShapes(mask, num_chanell = 1)

saveMask(RootPathFileMask, mask)

list_im_name = createListName(RootPathImages)
images = readAllIMG(RootPathImages, list_im_name)
images  = correctionShapes(images, num_chanell = 3)
saveImage(RootPathImages, images)

masks = readMask(RootPathFileMask)
images = readImages(RootPathImages)
saveUnionFile(RootPathFileMask, images, masks)
#images, masks = readUnionFile(RootPathFile)
#masks_one_hot = to_categorical(masks)

#plotforcheck(RootPathFile + '/TestPictures')