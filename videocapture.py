""" Utilites to launch webcam, capture/record video, show rectangles & text on screen.
"""

import time
import numpy as np
import cv2

from preprocess_utils import image_crop
from opticalflow import OpticalFlow, flow2colorimage

def video_start(device = 0, tuResolution =(320, 240), nFramePerSecond = 30):
	""" Returns videocapture object/stream
	Parameters:
		device: 0 for the primary webcam, 1 for attached webcam
	"""
	
	# try to open webcam device
	oStream = cv2.VideoCapture(device) 
	if not oStream.isOpened():
		# try again with inbuilt camera
		print("Try to initialize inbuilt camera ...")
		device = 0
		oStream = cv2.VideoCapture(device)
		if not oStream.isOpened(): raise ValueError("Could not open webcam")

	# set camera resolution
	nWidth, nHeight = tuResolution
	oStream.set(3, nWidth)
	oStream.set(4, nHeight)

	# try to set camera frame rate
	oStream.set(cv2.CAP_PROP_FPS, nFramePerSecond)

	print("Initialized video device %d, with resolution %s and target frame rate %d" % \
		(device, str(tuResolution), nFramePerSecond))

	return oStream

def rectangle_text(arImage, sColor, sUpper, sLower = None, tuRectangle = (224, 224)):
	""" Returns new image (not altering arImage)
	"""

	nHeigth, nWidth, _ = arImage.shape
	nRectHeigth, nRectWidth = tuRectangle
	x1 = int((nWidth - nRectWidth) / 2)
	y1 = int((nHeigth - nRectHeigth) / 2)

	if sColor == "green": bgr = (84, 175, 25)
	elif sColor == "orange": bgr = (60, 125, 235)
	else: #sColor == "red": 
		bgr = (27, 13, 252)

	arImageNew = np.copy(arImage)
	cv2.rectangle(arImageNew, (x1, y1), (nWidth-x1, nHeigth-y1), bgr, 3)

	# display a text to the frame 
	font = cv2.FONT_HERSHEY_SIMPLEX
	fFontSize = 0.5
	textSize = cv2.getTextSize(sUpper, font, 1.0, 2)[0]
	cv2.putText(arImageNew, sUpper, (x1 + 7, y1 + textSize[1] + 7), font, fFontSize, bgr, 2)	

	# 2nd text
	if (sLower != None):
		textSize = cv2.getTextSize(sLower, font, 1.0, 2)[0]
		cv2.putText(arImageNew, sLower, (x1 + 7, nHeigth - y1 - 7), font, fFontSize, bgr, 2)

	return arImageNew

def video_show(oStream, sColor, sUpper, sLower = None, tuRectangle = (224, 224), nCountdown = 0): 
	
	if nCountdown > 0: 
		fTimeTarget = time.time() + nCountdown
	
	# loop over frames from the video file stream
	s = sUpper
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		if bGrabbed == False: continue

		if nCountdown > 0:
			fCountdown = fTimeTarget - time.time()
			s = sUpper + str(int(fCountdown)+1) + " sec"

		# paint rectangle & text, show the (mirrored) frame
		arFrame = rectangle_text(cv2.flip(arFrame, 1), sColor, s, sLower, tuRectangle)
		cv2.imshow("Video", arFrame)
	
		# stop after countdown
		if nCountdown > 0 and fCountdown <= 0.0:
			key = -1
			break

		# Press 'q' to exit live loop
		key = cv2.waitKey(1) & 0xFF
		if key != 0xFF: break
	return key

def video_capture(oStream, sColor, sText, tuRectangle = (224, 224), nTimeDuration = 3, bOpticalFlow = False):
	
	if bOpticalFlow:
		oOpticalFlow = OpticalFlow(bThirdChannel = True)

	liFrames = []
	liFlows = []
	fTimeStart = time.time()

	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		arFrame = cv2.flip(arFrame, 1)
		liFrames.append(arFrame)

		fTimeElapsed = time.time() - fTimeStart
		s = sText + str(int(fTimeElapsed)+1) + " sec"

		# paint rectangle & text, show the frame
		arFrameText = rectangle_text(arFrame, sColor, s, "", tuRectangle)
		cv2.imshow("Video", arFrameText)

		# display optical flow
		if bOpticalFlow:
			arFlow = oOpticalFlow.next(image_crop(arFrame, *tuRectangle))
			liFlows.append(arFlow)
			cv2.imshow("Optical flow", flow2colorimage(arFlow))

		# stop after nTimeDuration sec
		if fTimeElapsed >= nTimeDuration: break

		# Press 'q' for early exit
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'): break
		cv2.waitKey(1)

	return fTimeElapsed, np.array(liFrames), np.array(liFlows)

def frame_show(oStream, sColor:str, sText:str, tuRectangle = (224, 224)):
	""" Read frame from webcam and display it with box+text """

	(_, oFrame) = oStream.read()
	oFrame = rectangle_text(cv2.flip(oFrame, 1), sColor, sText, "", tuRectangle)
	cv2.imshow("Video", oFrame)
	cv2.waitKey(1)

	return