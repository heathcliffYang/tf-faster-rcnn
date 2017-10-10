#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
import scipy.misc as misc

from utils.timer import Timer
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import os, cv2, time
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

#TODO: Use COCO dataset category
#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES= ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


#TODO: change the model to load
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_490000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
#DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
#TODO: change the dataset
DATASETS= {'coco_2014': ('coco_2014_train',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

color_list = ['#8E8E8E','#AAAAFF','#ff7575','#FF77FF','#CA8EFF','#84C1FF','#E1E100','#4F9D9D','#AE57A4']


FONT = ImageFont.truetype('UbuntuMono-BI.ttf', 55)

def vis_detections(im, class_name, dets, masks, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    draw = ImageDraw.Draw(im)
 
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        (left, right, top, bottom) = (bbox[0], bbox[2], bbox[1], bbox[3])

        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
		  width = 6, fill = 'red')
        text_top = top
	denote = class_name + ': {:.2f}'.format(dets[i,-1])
        text_width, text_height = FONT.getsize(denote)
        margin = np.ceil(0.05 * text_height)
        draw.text((left + margin, text_top - text_height - margin),
		 denote, fill= 'yellow', font= FONT)

        height, width = masks[i].shape

        masks[i] = masks[i].astype(np.uint8)

        for a in range(height):
            for b in range(width):
                if (masks[i,a,b] < 200):
                    masks[i,a,b] = 50
        
        mask_bit = np.zeros([height,width,3], dtype=np.uint8)
        
        y = np.expand_dims(np.invert(masks[i].astype(np.uint8)),axis=2)
        z = np.concatenate((mask_bit, y),axis=2)

        mask_resize = misc.imresize(z,(bottom-top,right-left,4))

        mask_bmp = Image.fromarray(mask_resize).convert("RGBA")
        
        if i >= 8:
            i = 0
        draw.bitmap((left,top), bitmap=mask_bmp, fill=color_list[i])
        print('Finish bounding boxes and masks drawing')

    del draw

def demo(sess, net, im_file, ind):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, masks = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    
    im = Image.open(im_file)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
    #TODO: Pick up target masks
        cls_masks = masks[:,:,:, cls_ind]

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, cls_masks, thresh=CONF_THRESH)

    im.save('/home/hugikun999/newtry-tf-faster-rcnn/tf-faster-rcnn/data/demo/result/result_{:d}.jpg'.format(ind))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    print(tfmodel)


    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 81,
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    print('saver successfully be created!!!')
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    ind = 0

    while True:
        try:
            #TODO : try : give fix name Showshow.jpg
            im_name = 'input_{:d}.jpg'.format(ind)
            im_file = os.path.join(cfg.DATA_DIR, 'demo', 'input', im_name)
            if os.path.isfile(im_file) == True:
                print('Demo for data/demo/{}'.format(im_name))
                demo(sess, net, im_file, ind)
                ind += 1
            else:
                raise Exception
        except Exception:
            continue
