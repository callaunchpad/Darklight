import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("/Users/zacharylieberman/desktop/DSC_0036.mov")
frameCount = 50
frameWidth = 1920
frameHeight = 1080

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()

cv2.namedWindow('frame 10')
for x in range(0):
   cv2.imshow('frame 10', buf[x])
    # plt.imshow(buf[x][:][0][:])
    # plt.show()
   cv2.waitKey(0)
