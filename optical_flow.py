import numpy as np
import cv2 as cv

cap = cv.VideoCapture(cv.samples.findFile("sample.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
frameNumber = 0

while(1):
    ret, frame2 = cap.read()
    if ret == False:
        break
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    # Farneback algorithm
    # flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 5, 3, 5, 1.2, 0) # takes time to compute

    # tv-l1 dual optical flow
    # optical_flow = cv.optflow.DualTVL1OpticalFlow_create(scaleStep = 0.5, warps = 3, epsilon = 0.02) # takes time to compute
    # tv-l1 dual optical flow - very fast
    # optical_flow = cv.optflow.DualTVL1OpticalFlow_create(theta = 0.1, nscales = 1, scaleStep = 0.5, warps = 1, epsilon = 0.1) # takes time to compute
    optical_flow = cv.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prvs, next, None)
    
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    
    # WRITE THE IMAGES
    # cv.imwrite(f'opticalhsv{frameNumber}.png', bgr)

    if k == 27:
        break
    prvs = next
    frameNumber = frameNumber + 1