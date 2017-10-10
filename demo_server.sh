#!/bin/bash

rm -f data/demo/input/*
rm -f data/demo/result/*

python tools/demo.py --net vgg16 --dataset coco_2014
