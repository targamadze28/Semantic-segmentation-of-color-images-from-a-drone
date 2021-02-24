#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:18:29 2020

@author: vsevolod
"""

import numpy as np
import h5py as h5
import os.path as fs
import matplotlib.pyplot as plt

from keras.utils import to_categorical

from numba import njit, prange, uint8, int32
from scipy.stats import t

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
def computationIndexClass(masks):
  n = len(masks)
  class_unique = np.unique(masks)
  indexes = np.zeros((n, len(class_unique)), dtype = np.uint8)
  for class_iter, class_id in enumerate(class_unique):
    for i in range(n):
      if class_id in masks[i]:
        indexes[i, class_iter] = True
  return indexes

def computationAreaClass(masks):
  n = len(masks)
  num_class = masks.shape[-1]
  class_area_on_sample = np.empty((n, num_class), dtype = np.float32)
  for sample_iter in range(n):
    for class_iter in range(num_class):
      class_area_on_sample[sample_iter, class_iter] = np.count_nonzero(masks[sample_iter, \
                                                    :, :, class_iter])
  return class_area_on_sample
#%%
RoothPath = '/home/vsevolod/Desktop/Dymas/Satellite'
DataPath = fs.join(RoothPath, 'NN/Data')
EDAPath = fs.join(RoothPath, 'EDA')
NameFile = 'Random150samples.hdf5'
#%%
false_numbers = [5, 9, 10, 20, 29, 40, 45, 70,\
                 108, 110, 120, 121, 122, 132, 143]
true_numbers = []
for i in range(150):
  if i not in false_numbers:
    true_numbers.append(i)

data = readHDF5file(DataPath, NameFile, ['Images', 'Masks'])
images, masks = data[0], data[1]

#images, masks = images[true_numbers], masks[true_numbers]

indexes = computationIndexClass(masks)
saveHDF5file(EDAPath, 'IndexClassEverySamples.hdf5', ['indexes_class'], [indexes])

one_hot_masks = to_categorical(masks)
areaonsample = computationAreaClass(one_hot_masks)
saveHDF5file(EDAPath, 'AreaClassEverySamples.hdf5', ['area_class'], [areaonsample])
#%%
indexesSourceDataSet = readHDF5file(EDAPath, 'IndexClassEverySamplesAugmentationTestDataSet.hdf5',\
                               ['indexes_class'])[0]
indexesCropDataSet = readHDF5file(EDAPath, 'IndexClassEverySamplesCropDataSet.hdf5',\
                               ['indexes_class'])[0]
areaSourceDataSet = readHDF5file(EDAPath, 'AreaClassEverySamplesSourceDataSet.hdf5',\
                               ['area_class'])[0]
areaCropDataSet = readHDF5file(EDAPath, 'AreaClassEverySamplesCropDataSet.hdf5',\
                               ['area_class'])[0]
num_class = 6
n_sourceDataSet = len(indexesSourceDataSet)
n_cropDataSet = len(indexesCropDataSet)

for i in range(num_class):
  sum_class = np.sum(indexesSourceDataSet[:, i])
  print('Частота появления %d класса в общем наборе данных %f'%(i,\
                                                                sum_class/n_sourceDataSet))
    
for i in range(num_class):
  sum_class = np.sum(indexesCropDataSet[:, i])
  print('Частота появления %d класса в кропнутом наборе данных %f'%(i,\
                                                                sum_class/n_cropDataSet))

for i in range(num_class):
  sum_class = np.sum(areaSourceDataSet[:, i])/np.sum(areaSourceDataSet)
  print('Относительная площадь %d класса в общем наборе данных %f'%(i,\
                                                                sum_class))
    
for i in range(num_class):
  sum_class = np.sum(areaCropDataSet[:, i])/(np.sum(areaCropDataSet))
  print('Относительная площадь %d класса в кропнутом наборе данных %f'%(i,\
                                                                sum_class))

for class_iter in range(num_class):
  S_on_class = 0
  for i in range(n_cropDataSet):
    if np.any([np.all([np.sum(indexesCropDataSet[i, :], axis = -1) == 1, indexesCropDataSet[i, class_iter] == 1]),\
               np.all([np.sum(indexesCropDataSet[i, :], axis = -1) == 2, indexesCropDataSet[i, class_iter] == 1, indexesCropDataSet[i, 0] == 1])]):
      S_on_class += 1
  print('Число семплов, в которых класс %d появляется один %d, %f'%(class_iter,\
                                                                S_on_class, S_on_class/n_cropDataSet))
#%%
"""
Построение распределений цветов для классов отдельно по каналам.
"""
@njit([int32[:, :, :, :](uint8[:, :, :, :], uint8[:, :, :, :])])
def computationPixelsOnClass(images, masks):
  n = len(masks)
  unique = np.unique(masks)
  pixels = np.zeros((n, 6, 160**2, 3), dtype = np.int32)
  pixels[:, :, :, :] = -1
  for class_iter in range(len(unique)):
    for i in range(n):
      args = np.argwhere(masks[i, :, :, 0] == unique[class_iter])
      if len(args) > 0:
        for indexes in range(len(args)):
          pixels[i, class_iter, indexes, :] = images[i, args[indexes, 0], args[indexes, 0], :]
  return pixels
    
def plotHists(pixels, PathToSave):
  colors = ['red', 'green', 'blue']
  for class_iter in range(6):
    for channel_iter in range(3):
      flt = pixels_on_class[:, class_iter, :, channel_iter].flatten()
      flt_rl = flt[np.argwhere(flt != -1)[:, 0]]
      fig = plt.figure(figsize = (10, 10))
      plt.hist(flt_rl, bins = 100, color = colors[channel_iter])
      plt.xlabel('color')
      plt.ylabel('num pixels')
      plt.title('color distribution class %d chanel %d'%(class_iter, channel_iter))
      plt.savefig(fs.join(PathToSave, 'cdcl%dch%d.png'%(class_iter, channel_iter)))
  return None

def dependent_ttest(data1, data2, alpha):
  mean1, mean2 = np.mean(data1), np.mean(data2)
  n = len(data1)
  d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
  d2 = sum([data1[i]-data2[i] for i in range(n)])
  sd = np.sqrt((d1 - (d2**2 / n)) / (n - 1))
  sed = sd / np.sqrt(n)
  t_stat = (mean1 - mean2) / sed
  df = n - 1
  cv = t.ppf(1.0 - alpha, df)
  p = (1.0 - t.cdf(np.abs(t_stat), df)) * 2.0
  return t_stat, df, cv, p

def tablecriterion():

NameFile = ''
data = readHDF5file(DataPath, NameFile, ['Images', 'Masks'])
images, masks = data[0], data[1]

PathToSave = '/home/vsevolod/Desktop/Dymas/Satellite/EDA/ColorDist'
pixels_on_class = computationPixelsOnClass(images, masks)
plotHists(pixels_on_class, PathToSave)
saveHDF5file(EDAPath, 'PixelsOnClass.hdf5', ['pixels'], [pixels_on_class])

#%%
def computationColorStatistic(images, indexesCropDataSet):
  means_all_class = []
  medians_all_class = []
  vars_all_class = []
  
  n_cropDataSet = len(indexesCropDataSet)
  num_class = 6
  for class_iter in range(num_class):
    ind_tmp = []
    for i in range(n_cropDataSet):
      if np.all([np.sum(indexesCropDataSet[i, :], axis = -1) == 1,
          indexesCropDataSet[i, class_iter] == 1]):
        ind_tmp.append(i)
    imgs_solo_class = images[ind_tmp]
    means = np.empty((len(imgs_solo_class), 3), dtype = np.float32)
    median = np.empty((len(imgs_solo_class), 3), dtype = np.float32)
    var = np.empty((len(imgs_solo_class), 3), dtype = np.float32)
    for imgs_iter, imgs in enumerate(imgs_solo_class):
      for kk in range(3):
        means[imgs_iter, kk] = np.mean(imgs[:, :, kk])
        median[imgs_iter] = np.median(imgs[:, :, kk])
        var[imgs_iter] = np.var(imgs[:, :, kk])
    means_all_class.append(np.mean(means, axis = -1)) 
    medians_all_class.append(np.mean(median, axis = -1))
    vars_all_class.append(np.mean(var, axis = -1))
  return means_all_class, medians_all_class, vars_all_class

def plotQuantile(stats, PathToSave, title = ''):
  percentile_list = [0, 0.5, 5, 25, 50, 75, 95, 99.5, 100]
  quantile_tmp = np.zeros((len(percentile_list), ), dtype = np.float32)
  num_class = len(stats)
  fig = plt.figure(figsize = (10, 10))
  for i in range(num_class):
    if len(stats[i]) > 0:
      print('start')
      for p_iter, p in enumerate(percentile_list):
        quantile_tmp[p_iter] = np.percentile(stats[i], p)
      plt.plot(percentile_list, quantile_tmp, label ='class %d'%(i))
  plt.xlabel('percentile')
  plt.ylabel('stats value')
  plt.legend()
  plt.title(title)
  plt.grid()
  plt.savefig(fs.join(PathToSave, title + '.png'))
  return None

def plotHistStats(stats, PathToSave, title = ''):
  fig, ax = plt.subplots(nrows=3, ncols=2)
  fig.set_figwidth(15)    #  ширина и
  fig.set_figheight(25)    #  высота "Figure"

  #fig.suptitle('Sample № %d'%(i), fontsize=10)
  ax[0, 0].hist(stats[0], bins = 100, color = 'b')
  ax[0, 0].set_title(title + ' class 0')

  ax[0, 1].hist(stats[1], bins = 100, color = 'b')
  ax[0, 1].set_title(title + ' class 1')

  ax[1, 0].hist(stats[2], bins = 100, color = 'b')
  ax[1, 0].set_title(title + ' class 2')

  ax[1, 1].hist(stats[3], bins = 100, color = 'b')
  ax[1, 1].set_title(title + ' class 3')

  ax[2, 0].hist(stats[4], bins = 100, color = 'b')
  ax[2, 0].set_title(title + ' class 4')
  
  ax[2, 1].hist(stats[5], bins = 100, color = 'b')
  ax[2, 1].set_title(title + ' class 5')
    
  fig.savefig(fs.join(PathToSave, title + '.png'))   
  return None
#%%
"""
Визуализация квантилей и распределений дисперсий цветов внутри классов
"""
PathToSave = '/home/vsevolod/Desktop/Dymas/Satellite/EDA/ColorDist'
NameFile = 'Random150samples.hdf5'
data = readHDF5file(DataPath, NameFile, ['Images', 'Masks'])
images, masks = data[0], data[1]

indexesCropDataSet = readHDF5file(EDAPath, 'IndexClassEverySamplesCropDataSet.hdf5',\
                               ['indexes_class'])[0]
indexesCropDataSet = indexesCropDataSet[:150]

means, medians, vars_ = computationColorStatistic(images, indexesCropDataSet)
plotQuantile(means, PathToSave, 'mean percentile')
plotQuantile(medians, PathToSave, 'median percentile')
plotQuantile(vars_, PathToSave, 'var percentile')

plotHistStats(means, PathToSave, 'mean hist')
plotHistStats(medians, PathToSave, 'median hist')
plotHistStats(vars_, PathToSave, 'var hist')
#%%
data_ind = readHDF5file('/home/vsevolod/Desktop/Dymas/Satellite/NN/Data',\
                        'SplitIndex.hdf5', ['train', 'test'])
train_ind_new, test_ind = data[0], data[1]
