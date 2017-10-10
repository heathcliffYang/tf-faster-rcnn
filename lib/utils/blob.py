# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import scipy.ndimage

def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
  """Mean subtract and scale an image for use in a blob."""
  im = im.astype(np.float32, copy=False)
  im -= pixel_means
  im_shape = im.shape

  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  im_scale = float(target_size) / float(im_size_min)
  # Prevent the biggest axis from being more than MAX_SIZE
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                  interpolation=cv2.INTER_LINEAR)

  return im, im_scale

def prep_mask_for_blob(gt_masks, im_info):
    shape = gt_masks.shape
    im_scale = im_info[2]
    im_info = np.ceil(im_info).astype(np.int32)
    final_mask = np.zeros([shape[0], im_info[0], im_info[1]])
   
    for i in range(shape[0]):
      gt_mask = np.array(gt_masks[i], dtype= np.float64)
      gt_mask = scipy.ndimage.zoom(gt_mask, im_scale, order=1)
      tmp_mask = np.zeros([im_info[0], im_info[1]])
      
      if gt_mask.shape[0] < im_info[0] and gt_mask.shape[1] < im_info[1]:
        tmp_mask[:gt_mask.shape[0], :gt_mask.shape[1]] = gt_mask
        tmp_mask[-1, :gt_mask.shape[1]] = gt_mask[-1,:]
        tmp_mask[:gt_mask.shape[0], -1] = gt_mask[:,-1]
      elif gt_mask.shape[0] < im_info[0]:
        tmp_mask[:gt_mask.shape[0], :] = gt_mask[:, :im_info[1]]
        tmp_mask[-1, :] = gt_mask[-1,:im_info[1]]
      elif gt_mask.shape[1] < im_info[1]: 
        tmp_mask[: , :gt_mask.shape[1]] = gt_mask[:im_info[0], :]
        tmp_mask[:, -1] = gt_mask[:,-1]
      else:
        tmp_mask = gt_mask[:im_info[0], :im_info[1]]

      final_mask[i] = tmp_mask
    
    return final_mask

