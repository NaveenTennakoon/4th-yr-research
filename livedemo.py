""" This module 
* launches the webcam, 
* waits for the start signal from user,
* captures 5 seconds of video,
* extracts frames from the video
* calculates the optical flow,
* and uses the neural network to predict the sign language gesture.
* Then start again.
"""

import os
import numpy as np
import cv2

from timer import Timer
from preprocess_utils import frames_downsample, images_crop
from videocapture import video_start, frame_show, video_show, video_capture
from optical_flow import frames2flows
from lip_extractor import bodyFrames2LipFrames
from datagenerator import VideoClasses
from model_i3d import I3D_load
from predict import probability2label

def livedemo(fused=False):

	diVideoSet = {
		"sName" : "signs",
        "nBaseClasses" : 5,   		# number of base classes
		"nExtendedClasses" : 12, 	# number of classes
        "framesNorm" : 40,    		# number of frames per video
        "fDurationAvg" : 3.0 		# seconds
	}

	# class files
	baseClassFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nExtendedClasses"])
	classFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nExtendedClasses"])
	
	print("\nStarting gesture recognition live demo ... ")
	print(os.getcwd())
	print(diVideoSet)
	
	# load label descriptions
	baseClasses = VideoClasses(baseClassFile)
	classes = VideoClasses(classFile)

	h, w = 220, 310

	if fused:
		sModelFile = "model/12-earlyfuse-tl-rc-full-best.h5"
		keI3D = I3D_load(sModelFile)

	else:
		sFlowModelFile = "model/12-oflow-tl-rc-full-best.h5"
		sLipModelFile = "model/12-lip-tl-full-best.h5"
	
		keI3Dbase = I3D_load(sFlowModelFile)
		keI3DLip = I3D_load(sLipModelFile)

	# open a pointer to the webcam video stream
	oStream = video_start(device = 1, tuResolution = (427, 240), nFramePerSecond = 30)

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

			# Downsample frames, extract lip frames and crop images
			arFrames = frames_downsample(arFrames, diVideoSet["framesNorm"])
			lipFrames = bodyFrames2LipFrames(arFrames)
			arFrames = images_crop(arFrames, h, w)
			
			# Translate frames to flows - these are already scaled between [-1.0, 1.0]
			print("Calculate optical flow on %d frames ..." % len(arFrames))
			timer.start()
			arFlows = frames2flows(arFrames, bThirdChannel = False, bShow = True)
			print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

			if fused:
				# predict video from fused model			
				print("Predict video with %s ..." % (keI3D.name))
				arXBody = np.expand_dims(arFlows, axis=0)
				arXLip = np.expand_dims(lipFrames, axis=0)
				arProbas = keI3D.predict([arXBody, arXLip], verbose = 1)[0]
				_, sLabel, fProba = probability2label(arProbas, classes, nTop = 3)
				sResults = "Sign with Fusion: %s (%.0f%%)" % (sLabel, fProba*100.)
				
				print(sResults)
			else:
				# predict video from flows			
				print("Predict video with %s ..." % (keI3Dbase.name))
				arXBase = np.expand_dims(arFlows, axis=0)
				arProbasBase = keI3Dbase.predict(arXBase, verbose = 1)[0]
				_, sLabelBase, fProbaBase = probability2label(arProbasBase, baseClasses, nTop = 3)
				sResultsBase = "Sign with Flow images: %s (%.0f%%)" % (sLabelBase, fProbaBase*100.)

				# predict video from lip images		
				print("Predict video with %s ..." % (keI3DLip.name))	
				arXLip = np.expand_dims(lipFrames, axis=0)
				arProbasLip = keI3DLip.predict(arXLip, verbose = 1)[0]
				_, sLabelLip, fProbaLip = probability2label(arProbasLip, classes, nTop = 3)
				sResultsLip = "Sign with Lip images: %s (%.0f%%)" % (sLabelLip, fProbaLip*100.)

				print(sResultsBase, sResultsLip)
			nCount += 1

		# quit
		elif key == ord('q'):
			break

	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return

if __name__ == '__main__':

	fused = True

	livedemo(fused)