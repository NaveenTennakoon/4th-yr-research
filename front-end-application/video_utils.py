""" Utilites to convert frames and show text on screen.
"""

import numpy as np
import cv2

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