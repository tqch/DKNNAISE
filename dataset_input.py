from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import os
import sys
version = sys.version_info

import numpy as np
import scipy.io as sio
from functools import reduce
import csv
from matplotlib import pyplot as plt
import cv2
import utilities
import json
from PIL import  Image



class CIFAR10Data(object):

    def __init__(self, config, seed=None):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'
        self.rng = np.random.RandomState(1) if seed is None else np.random.RandomState(seed)

        model_dir = config.model.output_dir
        path = config.data.path
        method = config.data.poison_method
        calieps = config.data.poison_eps
        clean = config.data.clean_label
        target = config.data.target_label
        position = config.data.position
        color = config.data.color
        num_training_examples = config.training.num_examples

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))
        
        cali_indice = np.array([i for i in range(0,50000,round(50000/calieps))])
        #poison_indices = self.rng.choice(cali_indices, calieps, replace=False)
        cali_images = np.zeros((calieps, 32, 32, 3))
        cali_labels = self.rng.randint(0,10, calieps)
        for i in range(calieps):
            cali_images[i] = train_images[cali_indice[i]]
            cali_labels[i] = train_labels[i]
        train_images = np.concatenate((train_images, cali_images), axis=0)
        train_labels = np.concatenate((train_labels, cali_labels), axis=0)
        train_images = np.delete(train_images, cali_indice, axis=0)
        train_labels = np.delete(train_labels, cali_indice, axis=0)


        train_indices = np.arange(len(train_images))
        eval_indices = np.arange(len(eval_images))
        cali_indices = np.arange(len(cali_images))
        
        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        removed_indices_file = os.path.join(model_dir, 'removed_inds.npy')
        if os.path.exists(removed_indices_file):
            removed = np.load(os.path.join(model_dir, 'removed_inds.npy'))
            train_indices = np.delete(train_indices, removed)

        #for debugging purpos
        np.save(os.path.join(model_dir, 'train_indices.npy'), train_indices)

        if config.model.per_im_std:
            train_images = self._per_im_std(train_images)
            eval_images = self._per_im_std(eval_images)
            
        self.train_data = DataSubset(train_images[train_indices], train_labels[train_indices])
        self.cali_data = DataSubset(cali_images[cali_indices], cali_labels[cali_indices])
        self.eval_data = DataSubset(eval_images[eval_indices], eval_labels[eval_indices],seed=seed)

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])

    @staticmethod      
    def _per_im_std(ims):
        split_ims = np.split(ims, ims.shape[0], axis=0)
        num_pixels = reduce(lambda x,y:x*y, list(split_ims[0].shape),1)
        for ii in range(len(split_ims)):
            curmean = np.mean(split_ims[ii],keepdims=True)
            split_ims[ii] = split_ims[ii] - curmean
            curstd = np.std(split_ims[ii],keepdims=False)
            adjustedstd = max(curstd, 1.0/np.sqrt(num_pixels))
            split_ims[ii] = split_ims[ii]/adjustedstd
        return np.concatenate(split_ims)

class DataSubset(object):
    def __init__(self, xs, ys, num_examples=None, seed=None):

        # self.rng = np.random.RandomState(1) if seed is None \
        #            else np.random.RandomState(seed)
        # # np.random.seed(99)
        # if num_examples:
        #     xs, ys = self._per_class_subsample(xs, ys, num_examples,
        #                                        rng=self.rng)
        if seed is not None:
            np.random.seed(99)
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        # np.random.seed(99)
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys




if __name__ == "__main__":


    config_dict = utilities.get_config('config_cifar.json')

    model_dir = config_dict['model']['output_dir']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(model_dir, 'config_cifar.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)
    RestrictedImagenet(config, seed=19233)