#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 17:43:45 2020

@author: vsevolod
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import os.path as fs

import cv2 as cv

def readUnionFile(PathToRead):
  ff = h5.File(fs.join(PathToRead, 'DataSet.hdf5'), 'r')
  images = ff['Images'][...]
  masks = ff['Masks'][...]
  ff.close()
  return images, masks

def saveCrops(PathToSave, images, masks):
  ff = h5.File(fs.join(PathToSave, 'CropDataSet.hdf5'), 'w')
  ff.create_dataset('Images', data = images)
  ff.create_dataset('Masks', data = masks)
  ff.close()
  return None

def make_prediction_cropped(data, initial_size=(240, 240), final_size=(230, 230),\
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
  tmp = make_prediction_cropped(images[0])
  k = len(tmp)
  masks = np.reshape(masks, masks.shape + (1,))
  crops_images = np.empty((k*n, 240, 240, 3), dtype = np.uint8)
  crops_masks = np.empty((k*n, 240, 240, 1), dtype = np.uint8)
  for i in range(n):
    crops_images[k*i:k*(i+1)] = make_prediction_cropped(images[i])
    crops_masks[k*i:k*(i+1)] = make_prediction_cropped(masks[i], num_channels = 1)
  return crops_images, crops_masks
  
RootPath2Dataset = '/home/vsevolod/Desktop/Dymas/Satellite/Data'

  
images, masks = readUnionFile(RootPath2Dataset)
crops_images, crops_masks = createCropDataSet(images, \
                                              masks)
print(crops_images.shape, crops_masks.shape)
saveCrops(RootPath2Dataset, crops_images, crops_masks)