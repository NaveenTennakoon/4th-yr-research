import argparse
import glob
import os
import cv2
import numpy as np

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/yolo/cross-hands.cfg", "models/yolo/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/yolo/cross-hands-tiny-prn.cfg", "models/yolo/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/yolo/cross-hands-yolov4-tiny.cfg", "models/yolo/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/yolo/cross-hands-tiny.cfg", "models/yolo/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("extracting tags for each image...")
if args.images.endswith(".txt"):
    with open(args.images, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(os.path.dirname(args.images), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % args.images))

conf_sum = 0
detection_count = 0

for file in files:
    mat = cv2.imread(file)
    width, height, inference_time, results = yolo.inference(mat)
    print("%s in %s seconds: %s classes found!" %(os.path.basename(file), round(inference_time, 2), len(results)))

    if(len(results)) < 2: continue
    
    print(results)
    left_hand = np.array(mat.copy())
    right_hand = np.array(mat.copy())

    cv2.namedWindow('left', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('left', 240, 240)
    cv2.namedWindow('right', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('right', 240, 240)

    counter = 0
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        if counter == 0: left_hand = left_hand[y:y+h, x:x+w, :]
        else: right_hand = right_hand[y:y+h, x:x+w, :]

        conf_sum += confidence
        detection_count += 1
        counter += 1

    # show the output image
    cv2.imshow('left', left_hand[:, :, :])
    cv2.imshow('right', right_hand[:, :, :])
    cv2.waitKey(0)

print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
cv2.destroyAllWindows()
