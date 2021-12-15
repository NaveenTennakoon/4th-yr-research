""" Utilites to process video inputs.
"""

import numpy as np
import cv2
import warnings
import tensorflow as tf
import menpo.io as mio
import glob
import os

from PIL import Image as PImage

from detect_face import create_detector, detect_face

# def get_stream(device = 0, resolution =(640, 360), fps = 30):
# 	""" Returns videocapture object/stream
# 	Parameters:
# 		device: 0 for the primary webcam, 1 for attached webcam
# 	"""

# 	# try to open webcam device
# 	stream = cv2.VideoCapture(device) 
# 	if not stream.isOpened():
# 		# try again with inbuilt camera
# 		print("Try to initialize inbuilt camera ...")
# 		device = 0
# 		stream = cv2.VideoCapture(device)
# 		if not stream.isOpened(): raise ValueError("Could not open webcam")

# 	# set camera resolution
# 	nWidth, nHeight = resolution
# 	stream.set(3, nWidth)
# 	stream.set(4, nHeight)

# 	# try to set camera frame rate
# 	stream.set(cv2.CAP_PROP_FPS, fps)

# 	print("Initialized video device %d, with resolution %s and target frame rate %d" % \
# 		(device, str(resolution), fps))

# 	return stream

def show_text(image, color, upper, lower = None):
	""" Returns new image with text
	"""

	height, _, _ = image.shape

	if color == "green": bgr = (84, 175, 25)
	elif color == "orange": bgr = (60, 125, 235)
	else:
		bgr = (27, 13, 252)

	image_new = np.copy(image)

	# display a text to the frame 
	font = cv2.FONT_HERSHEY_SIMPLEX
	fFontSize = 0.5
	textSize = cv2.getTextSize(upper, font, 1.0, 2)[0]
	cv2.putText(image_new, upper, (7, textSize[1] + 7), font, fFontSize, bgr, 2)	

	# 2nd text
	if (lower != None):
		textSize = cv2.getTextSize(lower, font, 1.0, 2)[0]
		cv2.putText(image_new, lower, (7, height - 7), font, fFontSize, bgr, 2)

	return image_new

def get_byte_image(frame):
    _, frame = cv2.imencode('.jpg', frame)
    
    return frame.tobytes()

def load_lip_detector():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = create_detector(sess, './face_detection')
    return pnet, rnet, onet

def bodyFrames2LipFrames(arFrames, pnet, rnet, onet):
    """ Extract lip image sequence from a given video(numpy array of frames)
    """
    minsize = 40
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709

    liFrames = []
    for nFrame in range(arFrames.shape[0]):
        # 3 step cascade detection 
        _, points = detect_face(arFrames[nFrame, :, :, :], 
                                            minsize, pnet, rnet, onet, threshold, factor)

        try:
            distances = (int((points[4, 0] - points[3, 0])/2), np.abs(int((points[9, 0] - points[8, 0])/2)))
            if(points[8, 0] < points[9, 0]):
                center = (int(points[3, 0]) + distances[0], int(points[8, 0]) + distances[1])
            else:
                center = (int(points[3, 0]) + distances[0], int(points[9, 0]) + distances[1])
            top_left = (center[0] - 8, center[1] - 8)
            bottom_right = (center[0] + 8, center[1] + 8)
            frame = arFrames[nFrame, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            frame = cv2.cvtColor(frame, 3)
            frame = cv2.resize(frame, dsize = (0,0), fx = 4, fy = 4, interpolation=cv2.INTER_LINEAR)
            liFrames.append(frame)
        except:
            print("Error during mouth segmentation")
            return None
    
    return np.array(liFrames)