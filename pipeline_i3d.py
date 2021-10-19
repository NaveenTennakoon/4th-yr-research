import os

from train_i3d import train_I3D_oflow_end2end, train_I3D_lipImage_end2end

if __name__ == '__main__':
    # Image and/or Optical flow pipeline
    bImage = True
    bOflow = False
    bLips = False

    diVideoSet = {
        "sName" : "slsl-22",
        "nClasses" : 22,        # number of classes
        "framesNorm" : 40,      # number of frames per video
        "nMinDim" : None,       # smaller dimension of saved video-frames
        "tuShape" : (720, 1280),# height, width
    }

    print("Starting I3D pipeline ...")
    print(os.getcwd())

    # train I3D network(s)
    if bOflow:
        train_I3D_oflow_end2end(diVideoSet)
    if bLips:
        train_I3D_lipImage_end2end(diVideoSet)
    elif bImage:
        raise ValueError("I3D training with only image data not implemented")