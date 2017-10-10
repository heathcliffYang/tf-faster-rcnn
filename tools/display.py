import cv2
import os
import time
import subprocess
import signal, sys

OUTPUT_DIR = '/home/hugikun999/newtry-tf-faster-rcnn/tf-faster-rcnn/data/demo/result/'
count = 0

name = 'result_0.jpg'
abs_name = OUTPUT_DIR + name
abs_name_old = OUTPUT_DIR + name
while(os.path.isfile(abs_name) == False):
    pass

while(True):
    
    if os.path.isfile(abs_name):
        im = cv2.imread(abs_name)
        if im is not None:
            if im.shape == (480,640,3):
                count += 1
                abs_name_old = abs_name
                name = 'result_{:d}.jpg'.format(count)
                abs_name = OUTPUT_DIR + name
            else:
                im = cv2.imread(abs_name_old)
        else:
            im = cv2.imread(abs_name_old)
    
    cv2.imshow('result', im)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
