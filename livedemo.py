""" This module 
* launches the webcam, 
* waits for the start signal from user,
* captures 5 seconds of video,
* extracts frames from the video
* calculates and displays the optical flow,
* and uses the neural network to predict the sign language gesture.
* Then start again.
"""

import os
import numpy as np
import cv2

from timer import Timer
from preprocess_utils import frames_downsample, images_crop
from videocapture import video_start, frame_show, video_show, video_capture
from opticalflow import frames2flows
from lip_extractor import bodyFrames2LipFrames
from datagenerator import VideoClasses
from model_i3d import I3D_load
from predict import probability2label

def livedemo():
	
	# dataset
	# diVideoSet = {"sName" : "signs",
    #     "nClasses" : 12,   # number of classes
    #     "framesNorm" : 40,    # number of frames per video
    #     "nMinDim" : 240,   # smaller dimension of saved video-frames
    #     "tuShape" : (720, 1280), # height, width
    #     "nFpsAvg" : 30,
    #     "nFramesAvg" : 90, 
    #     "fDurationAvg" : 3.0} # seconds

	diVideoSet = {"sName" : "signs",
        "nBaseClasses" : 5,   # number of base classes
		"nLipClasses" : 12, 	# number of classes
        "framesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShapeBase" : (720, 1280), # height, width
		"tuLipShape" : (112, 168), # height, width
        "nFpsAvg" : 30,
        "nFramesAvg" : 90, 
        "fDurationAvg" : 3.0} # seconds

	# files
	baseClassFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nBaseClasses"])
	lipClassFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nLipClasses"])
	
	print("\nStarting gesture recognition live demo ... ")
	print(os.getcwd())
	print(diVideoSet)
	
	# load label descriptions
	baseClasses = VideoClasses(baseClassFile)
	lipClasses = VideoClasses(lipClassFile)

	sFlowModelFile = "model/20210724-0915-signs005-oflow-i3d-above-best.h5"
	sLipModelFile = "model/20210723-0525-signs012-lips-i3d-entire-best.h5"
	h, w = 224, 224
	hL, wL = 112, 112
	keI3Dbase = I3D_load(sFlowModelFile, diVideoSet["framesNorm"], (h, w, 2), baseClasses.nClasses)
	keI3DLip = I3D_load(sLipModelFile, diVideoSet["framesNorm"], (hL, wL, 3), lipClasses.nClasses)

	# open a pointer to the webcam video stream
	oStream = video_start(device = 1, tuResolution = (427, 240), nFramePerSecond = 30)

	#liVideosDebug = glob.glob(sVideoDir + "/train/*/*.*")
	nCount = 0
	sResults = ""
	timer = Timer()

	# loop over action states
	while True:
		# show live video and wait for key stroke
		key = video_show(oStream, "green", "Press <blank> to start", sResults, tuRectangle = (h, w))
		
		# start!
		if key == ord(' '):
			# countdown n sec
			video_show(oStream, "orange", "Recording starts in ", tuRectangle = (h, w), nCountdown = 3)
			
			# record video for n sec
			fElapsed, arFrames, _ = video_capture(oStream, "red", "Recording ", \
				tuRectangle = (h, w), nTimeDuration = int(diVideoSet["fDurationAvg"]), bOpticalFlow = False)
			print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
				(fElapsed, str(arFrames.shape), len(arFrames)/fElapsed))

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...", tuRectangle = (h, w))

			# crop and downsample frames
			arFrames = frames_downsample(arFrames, diVideoSet["framesNorm"])
			
			lipFrames = bodyFrames2LipFrames(arFrames)
			arFrames = images_crop(arFrames, h, w)
			lipFrames = images_crop(lipFrames, hL, wL)

			# Translate frames to flows - these are already scaled between [-1.0, 1.0]
			print("Calculate optical flow on %d frames ..." % len(arFrames))
			timer.start()
			arFlows = frames2flows(arFrames, bThirdChannel = False, bShow = True)
			print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

			# predict video from flows			
			print("Predict video with %s ..." % (keI3Dbase.name))
			arX = np.expand_dims(arFlows, axis=0)
			arProbas = keI3Dbase.predict(arX, verbose = 1)[0]
			print(arProbas)
			_, sLabel, fProba = probability2label(arProbas, baseClasses, nTop = 3)

			arX = np.expand_dims(lipFrames, axis=0)
			arProbas = keI3DLip.predict(arX, verbose = 1)[0]
			print(arProbas)
			_, sLabel, fProba = probability2label(arProbas, lipClasses, nTop = 3)

			sResults = "Sign: %s (%.0f%%)" % (sLabel, fProba*100.)
			print(sResults)
			nCount += 1

		# quit
		elif key == ord('q'):
			break

	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return

if __name__ == '__main__':
	livedemo()