#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:49:52 2020

@author: vsevolod
"""

import numpy as np
import h5py as h5
import os.path as fs

import cv2 as cv

from scipy import ndimage

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
def createIndex4Augmentation(indexesCropDataSet):
  n = len(indexesCropDataSet)
  index_tmp = np.zeros((n, ), dtype = np.int32)
  for sample in range(n):
    if np.any([
        indexesCropDataSet[sample, 3] == 1,\
        indexesCropDataSet[sample, 4] == 1,\
        indexesCropDataSet[sample, 5] == 1]):
      index_tmp[sample] = 1
  return np.argwhere(index_tmp == 1)[:, 0]

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
#%%
RoothPath = '/data/vsevolod/DmiDiplom/SatelliteTask'
DataPath = fs.join(RoothPath, 'Data')
EDAPath = fs.join(RoothPath, 'EDA')
NameFile = 'DataSet.hdf5'
#%%
PathToSave = fs.join(EDAPath, 'Pictures')
NameFile = 'CropDataSet250x250.hdf5'
data = readHDF5file(DataPath, NameFile, ['Images', 'Masks'])
images, masks = data[0], data[1]

indexesCropDataSet = readHDF5file(EDAPath, 'IndexClassEverySamplesCropDataSet.hdf5',\
                               ['indexes_class'])[0]
#ind4aug = createIndex4Augmentation(indexesCropDataSet)
#saveHDF5file(EDAPath, 'Index4OfflineAugmentation.hdf5', ['index'], [ind4aug])

ind4aug = readHDF5file(EDAPath, 'Index4OfflineAugmentation.hdf5', ['index'])[0]
augimages, augmasks = offlineAugmentation(images[ind4aug])
images = np.concatenate((np.delete(images, ind4aug), augimages), axis = 0)
masks = np.concatenate((np.delete(masks, ind4aug), augmasks), axis = 0)
saveHDF5file(DataPath, 'CropDataSet250x250Augmentation.hdf5', ['Images', 'Masks'], [images, masks])
#%%
"""
Split on Train and Test data
"""
def findIndexesClass(all_index):
  num_class = 6
  indexes_every_class = []
  for class_iter in range(1, num_class):
    ind_class = np.argwhere(all_index[:, class_iter] == 1)[:, 0]
    indexes_every_class.append(ind_class)
  return indexes_every_class

def flatList(t):
  flat_list = []
  for sublist in t:
    for item in sublist:
      flat_list.append(item)
  return np.asarray(flat_list)

def splitIndex(indexes_every_class, percentage):
  n = len(indexes_every_class)
  train_index = []
  test_index = []
  for i in range(n):
    n_tmp = len(indexes_every_class[i])
    permut_index = indexes_every_class[np.random.permutation(n_tmp)]
    part = np.ceil(n_tmp*percentage)
    train_index.append(permut_index[part:])
    test_index.append(permut_index[:part])
  return flatList(train_index), flatList(test_index)

data = readHDF5file(DataPath, NameFile, ['Images', 'Masks'])
images, masks = data[0], data[1]

indexesAugmentDataSet = readHDF5file(EDAPath, 'IndexClassEverySamplesAugmentationDataSet.hdf5',\
                               ['indexes_class'])[0]

percentage = 0.2
indexes_every_class = findIndexesClass(indexesAugmentDataSet)
train_ind, test_ind = splitIndex(indexes_every_class, percentage)
print(len(train_ind))
print(len(test_ind))

saveHDF5file(DataPath, 'SplitIndex.hdf5', ['train', 'test'],\
             [train_ind, test_ind])

