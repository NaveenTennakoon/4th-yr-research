"""
This pipeline

* extracts frames/images from videos (training + validation)
* calculates optical flow
* trains an I3D neural network
"""

import os

from preprocess_utils import videosDir2framesDir
from opticalflow import framesDir2flowsDir
from train_i3d import train_I3D_oflow_end2end

if __name__ == '__main__':
    # Image and/or Optical flow pipeline
    bImage = False
    bOflow = True

    # dataset
    diVideoSet = {"sName" : "signs",
        "nClasses" : 12,   # number of classes
        "framesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (720, 1280), # height, width
        "nFpsAvg" : 30,
        "nFramesAvg" : 90, 
        "fDurationAvG" : 3.0} # seconds 

    # directories
    sFolder         = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    sClassFile      = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    videoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    imageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    imageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    oFlowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    oFlowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)

    print("Starting I3D pipeline ...")
    print(os.getcwd())

    # extract frames from videos
    if bImage or bOflow:
        print(videoDir)
        videosDir2framesDir(videoDir, imageDir, framesNorm = diVideoSet["framesNorm"],
            resizeMinDim = diVideoSet["nMinDim"], cropShape = None)

    # calculate optical flow
    if bOflow:
        framesDir2flowsDir(imageDir, oFlowDir, framesNorm = diVideoSet["framesNorm"])

    # train I3D network(s)
    if bOflow:
        train_I3D_oflow_end2end(diVideoSet)
    elif bImage:
        raise ValueError("I3D training with only image data not implemented")