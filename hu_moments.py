import cv2
from math import copysign, log10

from preprocess_utils import files2frames

frames = files2frames("data-temp/signs/012-40/image/train/c001/clothes_1")
first = cv2.cvtColor(frames[0, :, :, :], cv2.COLOR_BGR2GRAY)
_, prev = cv2.threshold(first, 128, 255, cv2.THRESH_BINARY)

for frame in frames:
    gry_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, next = cv2.threshold(gry_img, 128, 255, cv2.THRESH_BINARY)
    d2 = cv2.matchShapes(prev, next, cv2.CONTOURS_MATCH_I2,0) 
    prev = next
    print(d2)