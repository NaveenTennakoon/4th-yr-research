"""
Calculate optical flow from frames/images and save to disc
"""

import os
import glob
import warnings
import numpy as np
import cv2

from preprocess_utils import files2frames, frames_downsample

class OpticalFlow:
    """ Initialize an OpticalFlow object, 
    then use next() to calculate optical flow from subsequent frames.
    Detects first call automatically.
    """ 

    def __init__(self, algorithm:str = "tvl1-fast", bThirdChannel:bool = False, fBound:float = 20.):
        self.bThirdChannel = bThirdChannel
        self.fBound = fBound
        self.arPrev = np.zeros((1,1))

        if algorithm == "tvl1-fast":
            self.oTVL1 = cv2.optflow.DualTVL1OpticalFlow_create(
                scaleStep = 0.5, warps = 3, epsilon = 0.02)
                # Mo 25.6.2018: (theta = 0.1, nscales = 1, scaleStep = 0.3, warps = 4, epsilon = 0.02)
                # Very Fast (theta = 0.1, nscales = 1, scaleStep = 0.5, warps = 1, epsilon = 0.1)
            algorithm = "tvl1"

        elif algorithm == "tvl1-warps1":
            self.oTVL1 = cv2.optflow.DualTVL1OpticalFlow_create(warps = 1)
            algorithm = "tvl1"

        elif algorithm == "tvl1-quality":
            self.oTVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
                # Default: (tau=0.25, lambda=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01, 
                            #innnerIterations=30, outerIterations=10, scaleStep=0.8, gamma=0.0, 
                            #medianFiltering=5, useInitialFlow=False)
            algorithm = "tvl1"

        elif algorithm == "farneback":
            pass

        else: raise ValueError("Unknown optical flow type")
        
        self.algorithm = algorithm

        return

    def first(self, arImage:np.array) -> np.array:

        h, w, _ = arImage.shape

        # save first image in black&white
        self.arPrev = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)

        # first flow = zeros
        arFlow = np.zeros((h, w, 2), dtype = np.float32)

        if self.bThirdChannel:
            self.arZeros = np.zeros((h, w, 1), dtype = np.float32)
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2) 

        return arFlow

    def next(self, arImage:np.array) -> np.array:

        # first?
        if self.arPrev.shape == (1,1): return self.first(arImage)

        # get image in black&white
        arCurrent = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)

        if self.algorithm == "tvl1":
            arFlow = self.oTVL1.calc(self.arPrev, arCurrent, None)
        elif self.algorithm == "farneback":
            arFlow = cv2.calcOpticalFlowFarneback(self.arPrev, arCurrent, flow=None, 
                pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        else: raise ValueError("Unknown optical flow type")

        # only 2 channels
        arFlow = arFlow[:, :, 0:2]

        # truncate to +/-fBound, then rescale to [-1.0, 1.0]
        arFlow[arFlow > self.fBound] = self.fBound 
        arFlow[arFlow < -self.fBound] = -self.fBound
        arFlow = arFlow / self.fBound

        if self.bThirdChannel:
            # add third empty channel
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2) 

        self.arPrev = arCurrent

        return arFlow

def frames2flows(arFrames:np.array(int), algorithm = "tvl1-fast", bThirdChannel:bool = False, bShow = False, fBound:float = 20.) -> np.array(float):
    """ Calculates optical flow from frames

    Returns:
        array of flow-arrays, each with dim (h, w, 2), 
        with "flow"-values truncated to [-15.0, 15.0] and then scaled to [-1.0, 1.0]
        If bThirdChannel = True a third channel with zeros is added
    """

    # initialize optical flow calculation
    oOpticalFlow = OpticalFlow(algorithm = algorithm, bThirdChannel = bThirdChannel, fBound = fBound)
    
    liFlows = []
    # loop through all frames
    for i in range(len(arFrames)):
        # calc dense optical flow
        arFlow = oOpticalFlow.next(arFrames[i, ...])
        liFlows.append(arFlow)
        if bShow:
            cv2.imshow("Optical flow", flow2colorimage(arFlow))
            cv2.waitKey(1)

    return np.array(liFlows)

def flows_add_third_channel(arFlows:np.array) -> np.array:
    """ add third empty channel to array of flows
    """
    
    n, h, w, c = arFlows.shape
    if c != 2: raise ValueError("Expected 2 channels, not %d" % c)
    
    arZeros = np.zeros((n, h, w, 1), dtype = np.float32)
    arFlows3 = np.concatenate((arFlows, arZeros), axis=3)

    return arFlows3

def flows2files(arFlows:np.array(float), targetDir:str):
    """ Save array of flows (2 channels with values in [-1.0, 1.0]) 
    to jpg files (with 3 channels 0-255 each) in targetDir
    """

    n, h, w, _ = arFlows.shape
    os.makedirs(targetDir, exist_ok=True)
    arZeros = np.zeros((h, w, 1), dtype = np.float32)

    for i in range(n):
        # add third empty channel
        ar_f_Flow = np.concatenate((arFlows[i, ...], arZeros), axis=2)

        # rescale to 0-255  
        ar_n_Flow = np.round((ar_f_Flow + 1.0) * 127.5).astype(np.uint8)

        cv2.imwrite(targetDir + "/flow%03d.jpg"%(i), ar_n_Flow)

    return

def files2flows(sDir:str, b3channels:bool = False) -> np.array:
    """ Read flow files from directory
    Expects 3-channel jpg files
    Output
        Default: array with 2-channel flow, with floats between [-1.0, 1.0]
        If b3channels = True: including 3rd channel from jpeg (should result in zero values)
    """

    # important to sort flow files upfront
    liFiles = sorted(glob.glob(sDir + "/*.jpg"))

    if len(liFiles) == 0: raise ValueError("No optical flow files found in " + sDir)

    liFlows = []
    # loop through frames
    for i in range(len(liFiles)):

        ar_n_Flow = cv2.imread(liFiles[i])

        # optical flow only 2-dim
        if not b3channels: 
            ar_n_Flow = ar_n_Flow[:,:,0:2]

        # rescale from 0-255 to [-1.0, 1.0]
        ar_f_Flow = ((ar_n_Flow / 127.5) - 1.).astype(np.float32)
        
        liFlows.append(ar_f_Flow)

    return np.array(liFlows)

def flow2colorimage(ar_f_Flow:np.array(float)) -> np.array(int):
    """ translate 1 optical flow (with values from -1.0 to 1.0) to a color image
    """

    h, w, _ = ar_f_Flow.shape
    if not isinstance(ar_f_Flow[0,0,0], np.float32): 
        warnings.warn("Need to convert flows to float32")
        ar_f_Flow = ar_f_Flow.astype(np.float32)

    ar_n_hsv = np.zeros((h, w, 3), dtype = np.uint8)
    ar_n_hsv[...,1] = 255

    # get colors
    mag, ang = cv2.cartToPolar(ar_f_Flow[..., 0], ar_f_Flow[..., 1])
    ar_n_hsv[...,0] = ang * 180 / np.pi / 2
    ar_n_hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    ar_n_bgr = cv2.cvtColor(ar_n_hsv, cv2.COLOR_HSV2BGR)

    return ar_n_bgr

def flows2colorimages(arFlows:np.array) -> np.array:
    """ translate many optical flows to colorful images
    """

    n, _, _, _ = arFlows.shape
    liImages = []
    for i in range(n):
        arImage = flow2colorimage(arFlows[i, ...])
        liImages.append(arImage)

    return np.array(liImages)

def framesDir2flowsDir(framesBaseDir:str, flowBaseDir:str, framesNorm:int = None, algorithm:str = "tvl1-fast"):
    """ Calculate optical flow from frames (extracted from videos) 
    
    Input videoframe structure: ... frameDir / train / class001 / videoname / frames.jpg
    Output: ... flowDir / train / class001 / videoname / flow.jpg
    """

    # get list of directories with frames: ... / frameDir/train/class/videodir/frames.jpg
    currentDir = os.getcwd()
    os.chdir(framesBaseDir)
    videos = sorted(glob.glob("*/*/*"))
    os.chdir(currentDir)
    print("Found %d directories=videos with frames in %s" % (len(videos), framesBaseDir))

    # loop over all videos-directories
    counter = 0
    for frameDir in videos:

        # generate target directory
        flowDir = flowBaseDir + "/" + frameDir

        if framesNorm != None and os.path.exists(flowDir):
            nFlows = len(glob.glob(flowDir + "/*.*"))
            if nFlows == framesNorm: 
                print("Video %5d: optical flow already extracted to %s" % (counter, flowDir))
                counter += 1
                continue
            else: 
                print("Video %5d: Directory with %d instead of %d flows detected" % (counter, nFlows, framesNorm))

        # retrieve frame files - in ascending order
        arFrames = files2frames(framesBaseDir + "/" + frameDir)

        # downsample
        if framesNorm != None: 
            arFrames = frames_downsample(arFrames, framesNorm)

        # calculate and save optical flow
        print("Video %5d: Calc optical flow with %s from %s frames to %s" % (counter, algorithm, str(arFrames.shape), flowDir))
        arFlows = frames2flows(arFrames, algorithm = algorithm)
        flows2files(arFlows, flowDir)

        counter += 1      

    return