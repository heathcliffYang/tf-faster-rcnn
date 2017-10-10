import numpy as np
import cv2
import os
import time
import skvideo.io

OUTPUT_DIR = '/home/hugikun999/newtry-tf-faster-rcnn/tf-faster-rcnn/data/demo/input'

count = 1


#cap = skvideo.io.VideoCapture()
cap = cv2.VideoCapture(0)
#cap.open(0)

print(cap.isOpened())
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[:, ::-1, :]

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    if count % 10 == 0 or count == 1:
        name = 'input_{:d}.jpg'.format(count / 10)
        output = OUTPUT_DIR + '/' + name
        cv2.imwrite(OUTPUT_DIR + '/' + name, frame, [1,10])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    count += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
