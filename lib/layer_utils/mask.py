from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2

from utils.cython_bbox import bbox_overlaps
from model.config import cfg
from model.bbox_transform import bbox_transform

def encode(gt_masks, gt_boxes, rois, num_classes, mask_height, mask_width, im_height, im_width):
  """Encode masks groundtruth into learnable targets
  Sample some exmaples
  
  Params
  ------
  gt_masks: image_height x image_width {0, 1} matrix, of shape (G, imh, imw)
  gt_boxes: ground-truth boxes of shape (G, 5), each raw is [x1, y1, x2, y2, class]
  rois:     the bounding boxes of shape (N, 4), [x1, y1, x2, y2]
  ## scores:   scores of shape (N, 1)
  num_classes; K
  mask_height, mask_width: height and width of output masks
  
  Returns
  -------
  # rois: boxes sampled for cropping masks, of shape (M, 4)
  labels: class-ids of shape (M, 1)
  mask_targets: learning targets of shape (M, pooled_height, pooled_width, K) in {0, 1} values
  mask_inside_weights: of shape (M, pooled_height, pooled_width, K) in {0, 1} indicating which mask is sampled
  """
  total_mask = rois.shape[0]
  if gt_boxes.size > 0:
    
    overlaps = bbox_overlaps(
      np.ascontiguousarray(rois[:, 1:], dtype= np.float),
      np.ascontiguousarray(gt_boxes[:, :4], dtype= np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = np.zeros((total_mask, ), np.float32)
    labels[:] = -1

    keep_inds = np.where(max_overlaps >= cfg.TRAIN.mask_threshold)[0]
    num_masks = int(min(keep_inds.size, cfg.TRAIN.masks_per_image))
    if keep_inds.size > 0 and num_masks < keep_inds.size:
      keep_inds = np.random.choice(keep_inds, size=num_masks, replace=False)
    labels[keep_inds] = gt_boxes[gt_assignment[keep_inds], -1]

    ignore_inds = np.where(max_overlaps < cfg.TRAIN.fg_threshold)[0]
    labels[ignore_inds] = -1

    mask_targets = np.zeros((total_mask, mask_height, mask_width, num_classes), dtype=np.int32)
    mask_inside_weights = np.zeros(mask_targets.shape, dtype=np.float32)
    rois[rois < 0] = 0
    rois[rois[:, 3] > im_width - 1] = im_width - 1
    rois[rois[:, 4] > im_height - 1] = im_height - 1 
    
    for i in keep_inds:
      roi = rois[i, 1:]
      cropped = gt_masks[gt_assignment[i], int(roi[1]):int(roi[3])+1, int(roi[0]):int(roi[2])+1]
      cropped = cv2.resize(cropped, (mask_width, mask_height), interpolation=cv2.INTER_NEAREST)
      mask_targets[i, :, :, int(labels[i])] = cropped
      mask_inside_weights[i, :, :, int(labels[i])] = 1
  else: 
    labels = np.zeros((total_mask,), dtype=np.int32)
    labels[:] = -1
    mask_targets = np.zeros((total_mask, mask_height, mask_width, num_classes), dtype=np.int32)
    mask_inside_weights = np.zeros(mask_targets.shape, dtype=np.float32)

  return labels, mask_targets, mask_inside_weights

