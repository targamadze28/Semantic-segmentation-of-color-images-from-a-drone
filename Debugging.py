#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 23:50:13 2020

@author: vsevolod
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import os.path as fs
import cv2 as cv

from keras.utils import to_categorical
from scipy import ndimage


def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data

def make_prediction_cropped(data, initial_size=(2000, 2000), final_size=(1500, 1500),\
                             num_channels=3):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height = data.shape[0]
    width = data.shape[1]

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1
        
    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((padded_height, padded_width, num_channels), dtype = np.uint8)

    padded[shift:shift + height, shift: shift + width, :] = data
    
    # add mirror reflections to the padded areas
    up = padded[shift:2 * shift, shift:-shift, :][::-1, :]
    padded[:shift, shift:-shift, :] = up

    lag = padded.shape[0] - height - shift
    bottom = padded[height + shift - lag:shift + height, shift:-shift, :][::-1, :]
    padded[height + shift:, shift:-shift, :] = bottom

    left = padded[:, shift:2 * shift, :][:, ::-1, :]
    padded[:, :shift, :] = left

    lag = padded.shape[1] - width - shift
    right = padded[:, width + shift - lag:shift + width, :][:, ::-1, :]

    padded[:, width + shift:, :] = right

    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[h:h + initial_size[0], w:w + initial_size[0], :]]
            
    return np.asarray(temp)

def createCropDataSet(images, masks):
  n = len(images)
  print(n)
  tmp = make_prediction_cropped(images[0])
  k = len(tmp)
  masks = np.reshape(masks, masks.shape + (1,))
  crops_images = np.empty((k*n, 2000, 2000, 3), dtype = np.uint8)
  crops_masks = np.empty((k*n, 2000, 2000, 1), dtype = np.uint8)
  for i in range(n):
    crops_images[k*i:k*(i+1)] = make_prediction_cropped(images[i])
    crops_masks[k*i:k*(i+1)] = make_prediction_cropped(masks[i], num_channels = 1)
  return crops_images, crops_masks

def rotateAugmentation(images):
  n = len(images)
  rotateimages = np.empty((4*n, images.shape[1], images.shape[2], images.shape[3]),\
                          dtype = images.dtype)
  for i in range(n):
    rotateimages[4*i] = ndimage.rotate(images[i], 0)
    rotateimages[4*i + 1] = ndimage.rotate(images[i], 90)
    rotateimages[4*i + 2] = ndimage.rotate(images[i], 180)
    rotateimages[4*i + 3] = ndimage.rotate(images[i], 270)
  return rotateimages

def reflectionAugmentation(images):
  n = len(images)
  reflectionimages = np.empty((4*n, images.shape[1], images.shape[2], images.shape[3]),\
                          dtype = images.dtype)
  for i in range(n):
    reflectionimages[4*i] = images[i]
    reflectionimages[4*i + 1] = cv.flip(images[i], 0)
    reflectionimages[4*i + 2] = cv.flip(images[i], 1)
    reflectionimages[4*i + 3] = cv.flip(images[i], -1)
  return reflectionimages

def offlineAugmentation(images, masks):
  rotateimages = rotateAugmentation(images)
  reflectionimages = rotateAugmentation(rotateimages)
  rotatemasks = rotateAugmentation(masks)
  reflectionmasks = rotateAugmentation(rotatemasks)
  return reflectionimages, reflectionmasks

#DSC00021.hdf5
#DSC00021.JPG

PathToMask = '/home/vsevolod/Desktop/Dymas/Satellite/DataMarkII_'
NameMask = 'DSC00021.hdf5'

PathToImage = '/home/vsevolod/Documents/SatelliteImage/srcphoto7z/ExpDS'
NameImage = 'DSC00021.JPG'


mask = readHDF5file(PathToMask, NameMask, ['Mask'])[0]
image = cv.imread(fs.join(PathToImage, NameImage))
masks_one_hot = to_categorical(mask, num_classes = 6)

fig, ax = plt.subplots(nrows=4, ncols=2)
fig.set_figwidth(15)    #  ширина и
fig.set_figheight(35)    #  высота "Figure"

ax[0, 0].imshow(image)
ax[0, 0].set_title('source images')

ax[0, 1].imshow(masks_one_hot[:, :, 0])
ax[0, 1].set_title('class 0')

ax[1, 0].imshow(masks_one_hot[:, :, 1])
ax[1, 0].set_title('class 1')

ax[1, 1].imshow(masks_one_hot[:, :, 2])
ax[1, 1].set_title('class 2')

ax[2, 0].imshow(masks_one_hot[:, :, 3])
ax[2, 0].set_title('class 3')
  
ax[2, 1].imshow(masks_one_hot[:, :, 4])
ax[2, 1].set_title('class 4')

ax[3, 0].imshow(masks_one_hot[:, :, 5])
ax[3, 0].set_title('class 5')

plt.show()


image = np.reshape(image, ((1, ) + image.shape))
mask = np.reshape(mask, ((1, ) + mask.shape))

crops_images = make_prediction_cropped(image,\
                      initial_size=(2000, 2000), final_size=(1500, 1500))
  
crops_mask = make_prediction_cropped(mask,\
                      initial_size=(2000, 2000), final_size=(1500, 1500), num_channels = 1)

plt.imshow(image)
plt.show()

crops_images, crops_masks = createCropDataSet(image, mask)
crops_images, crops_masks = crops_images[:3], crops_masks[:3]

reflectionimages, reflectionmasks = offlineAugmentation(crops_images, crops_masks)

plt.imshow(image)
plt.show()

masks_one_hot = to_categorical(crops_masks[0], num_classes = 6)
fig, ax = plt.subplots(nrows=4, ncols=2)
fig.set_figwidth(15)    #  ширина и
fig.set_figheight(35)    #  высота "Figure"


ax[0, 0].imshow(crops_images[0])
ax[0, 0].set_title('source images')

ax[0, 1].imshow(masks_one_hot[:, :, 0])
ax[0, 1].set_title('class 0')

ax[1, 0].imshow(masks_one_hot[:, :, 1])
ax[1, 0].set_title('class 1')

ax[1, 1].imshow(masks_one_hot[:, :, 2])
ax[1, 1].set_title('class 2')

ax[2, 0].imshow(masks_one_hot[:, :, 3])
ax[2, 0].set_title('class 3')
  
ax[2, 1].imshow(masks_one_hot[:, :, 4])
ax[2, 1].set_title('class 4')

ax[3, 0].imshow(masks_one_hot[:, :, 5])
ax[3, 0].set_title('class 5')

plt.show()

masks_one_hot = to_categorical(crops_masks[0], num_classes = 6)
fig, ax = plt.subplots(nrows=4, ncols=2)
fig.set_figwidth(15)    #  ширина и
fig.set_figheight(35)    #  высота "Figure"

ax[0, 0].imshow(crops_images[0])
ax[0, 0].set_title('source images')

ax[0, 1].imshow(masks_one_hot[:, :, 0])  
ax[0, 1].set_title('class 0')

ax[1, 0].imshow(masks_one_hot[:, :, 1])
ax[1, 0].set_title('class 1')

ax[1, 1].imshow(masks_one_hot[:, :, 2])
ax[1, 1].set_title('class 2')

ax[2, 0].imshow(masks_one_hot[:, :, 3])
ax[2, 0].set_title('class 3')
  
ax[2, 1].imshow(masks_one_hot[:, :, 4])
ax[2, 1].set_title('class 4')

ax[3, 0].imshow(masks_one_hot[:, :, 5])
ax[3, 0].set_title('class 5')
plt.show()


for i in range(16):
  masks_one_hot = to_categorical(reflectionmasks[i], num_classes = 6)
  fig, ax = plt.subplots(nrows=4, ncols=2)
  fig.set_figwidth(15)    #  ширина и
  fig.set_figheight(35)    #  высота "Figure"


  ax[0, 0].imshow(reflectionimages[i])
  ax[0, 0].set_title('source images')

  ax[0, 1].imshow(masks_one_hot[:, :, 0])  
  ax[0, 1].set_title('class 0')

  ax[1, 0].imshow(masks_one_hot[:, :, 1])
  ax[1, 0].set_title('class 1')

  ax[1, 1].imshow(masks_one_hot[:, :, 2])
  ax[1, 1].set_title('class 2')

  ax[2, 0].imshow(masks_one_hot[:, :, 3])
  ax[2, 0].set_title('class 3')
  
  ax[2, 1].imshow(masks_one_hot[:, :, 4])
  ax[2, 1].set_title('class 4')

  ax[3, 0].imshow(masks_one_hot[:, :, 5])
  ax[3, 0].set_title('class 5')

  plt.show()
