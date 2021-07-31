import os

from preprocess_utils import videosDir2framesDir
from optical_flow import framesDir2flowsDir
from lip_extractor import bodyFramesDir2lipFrameDir
from train_i3d import train_I3D_oflow_end2end, train_I3D_lipImage_end2end

if __name__ == '__main__':
    # Image and/or Optical flow pipeline
    bImage = False
    bOflow = True
    bLips = False

    diVideoSet = {
        "sName" : "signs",
        "nClasses" : 12,        # number of classes
        "framesNorm" : 40,      # number of frames per video
        "nMinDim" : 240,        # smaller dimension of saved video-frames
        "tuShape" : (720, 1280),# height, width
    }
    
    # directories
    sFolder         = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    sClassFile      = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    videoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    orgImageDir     = "data-temp/%s/%s/org_image"%(diVideoSet["sName"], sFolder)
    imageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    imageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    oFlowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    lipDir          = "data-temp/%s/%s/lips"%(diVideoSet["sName"], sFolder)
    oFlowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)

    print("Starting I3D pipeline ...")
    print(os.getcwd())

    # extract frames from videos
    if bImage or bOflow or bLips:
        print(videoDir)
        videosDir2framesDir(videoDir, imageDir, framesNorm = diVideoSet["framesNorm"],
            resizeMinDim = diVideoSet["nMinDim"], cropShape = None)
    # calculate optical flow
    if bOflow:
        framesDir2flowsDir(imageDir, oFlowDir, framesNorm = diVideoSet["framesNorm"])
    if bLips:
        bodyFramesDir2lipFrameDir(imageDir, lipDir, minsize = 40, threshold = [ 0.6, 0.7, 0.7 ],
        factor = 0.709, grayscale=False, binary=False)

    # train I3D network(s)
    if bOflow:
        train_I3D_oflow_end2end(diVideoSet)
    if bLips:
        train_I3D_lipImage_end2end(diVideoSet)
    elif bImage:
        raise ValueError("I3D training with only image data not implemented")