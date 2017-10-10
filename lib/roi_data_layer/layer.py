# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import roi_data_layer.roidb as rdl_roidb #TODO
import numpy as np
import time

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""
  #TODO delete self._random
  def __init__(self, imdb, num_classes, random=False):
    """Set the roidb to be used by this layer during training."""
    self._num_classes = num_classes
    self._imdb = imdb
    #self._random = random
      
  def forward(self):
    """Get blobs and copy them into this layer's top blob vector."""
    minibatch_db = self._get_training_roidb()
    blobs = get_minibatch(minibatch_db, self._num_classes)
    return blobs
 
  def _get_training_roidb(self):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    roidb = rdl_roidb.prepare_roidb(self._imdb)
    roidb = self._filter_roidb(roidb)
    print('Preparing training data...{:d} done'.format(self._imdb.roidb_index()))

    return roidb

  def _filter_roidb(self, roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
      overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
      fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
      bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
      valid = len(fg_inds) > 0 or len(bg_inds) > 0
      return valid

    if is_valid(roidb):
      filtered_roidb = roidb
    else:
      print('This roidb is not valid!(load before)')
      filtered_roidb = self._get_training_roidb()
      print('This roidb is not valid!(load after)')

    return filtered_roidb
