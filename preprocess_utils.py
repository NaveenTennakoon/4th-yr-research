"""
Extract frames from a video (or many videos). 
Plus some frame/image manipulation utilities.
"""

import os
import glob
import warnings
import subprocess
import numpy as np
import pandas as pd
import cv2

def image_resize_aspectratio(arImage: np.array, minDim:int = 256) -> np.array:
    nHeigth, nWidth, _ = arImage.shape

    if nWidth >= nHeigth:
        # wider than high => map heigth to 224
        fRatio = minDim / nHeigth
    else: 
        fRatio = minDim / nWidth

    if fRatio != 1.0:
        arImage = cv2.resize(arImage, dsize = (0,0), fx = fRatio, fy = fRatio, interpolation=cv2.INTER_LINEAR)

    return arImage

def images_resize_aspectratio(arImages: np.array, minDim:int = 256) -> np.array:
    nImages, _, _, _ = arImages.shape
    liImages = []
    for i in range(nImages):
        arImage = image_resize_aspectratio(arImages[i, ...], minDim)
        liImages.append(arImage)
    return np.array(liImages)

def video2frames(sVideoPath:str, resizeMinDim:int) -> np.array:
    """ Read video file with OpenCV and return array of frames

    if nMinDim != None: Frames are resized preserving aspect ratio 
        so that the smallest dimension is eg 256 pixels, with bilinear interpolation
    """
  
    # Create a VideoCapture object and read from input file
    oVideo = cv2.VideoCapture(sVideoPath)
    if (oVideo.isOpened() == False): raise ValueError("Error opening video file")

    liFrames = []

    # Read until video is completed
    while(True):
        
        (bGrabbed, arFrame) = oVideo.read()
        if bGrabbed == False: break

        if resizeMinDim != None:
            # resize image
            arFrame = image_resize_aspectratio(arFrame, resizeMinDim)

		# Save the resulting frame to list
        liFrames.append(arFrame)
   
    return np.array(liFrames)

def frames2files(arFrames:np.array, sTargetDir:str):
    """ Write array of frames to jpg files
    Input: arFrames = (number of frames, height, width, depth)
    """

    os.makedirs(sTargetDir, exist_ok=True)
    for nFrame in range(arFrames.shape[0]):
        cv2.imwrite(sTargetDir + "/frame%04d.jpg" % nFrame, arFrames[nFrame, :, :, :])

    return

def files2frames(path:str) -> np.array:
    # important to sort image files upfront
    liFiles = sorted(glob.glob(path + "/*.jpg"))
    if len(liFiles) == 0: raise ValueError("No frames found in " + path)

    liFrames = []
    # loop through frames
    for frame in liFiles:
        arFrame = cv2.imread(frame)
        liFrames.append(arFrame)

    return np.array(liFrames)   
    
def frames_downsample(arFrames:np.array, targetNum:int) -> np.array:
    """ Adjust number of frames (eg 123) to targetNum (eg 79)
    works also if originally less frames then targetNum
    """

    nSamples, _, _, _ = arFrames.shape
    if nSamples == targetNum: return arFrames

    # down/upsample the list of frames
    fraction = nSamples / targetNum
    index = [int(fraction * i) for i in range(targetNum)]
    targetFrames = [arFrames[i,:,:,:] for i in index]

    return np.array(targetFrames)   
    
def image_crop(arFrame:np.array, targetHeight:int, targetWidth:int) -> np.array:
    """ crop 1 frame to specified size, choose centered image
    """

    height, width, _ = arFrame.shape

    if (height < targetHeight) or (width < targetWidth):
        raise ValueError("Image height/width too small to crop to target size")

    # calc left upper corner of target image
    sX = int(width/2 - targetWidth/2)
    sY = int(height/2 - targetHeight/2)

    arFrame = arFrame[sY:sY+targetHeight, sX:sX+targetWidth, :]

    return arFrame

def images_crop(arFrames:np.array, targetHeight:int, targetWidth:int) -> np.array:
    """ crop each frame in array to specified size, choose centered image
    """

    _, height, width, _ = arFrames.shape

    if (height < targetHeight) or (width < targetWidth):
        raise ValueError("Image height/width too small to crop to target size")

    # calc left upper corner
    sX = int(width/2 - targetWidth/2)
    sY = int(height/2 - targetHeight/2)

    arFrames = arFrames[:, sY:sY+targetHeight, sX:sX+targetWidth, :]

    return arFrames

def images_rescale(arFrames:np.array) -> np.array(float):
    """ Rescale array of images (rgb 0-255) to [-1.0, 1.0]
    """

    ar_fFrames = arFrames / 127.5
    ar_fFrames -= 1.

    return ar_fFrames

def images_normalize(arFrames:np.array, targetNum:int, height:int, width:int, rescale:bool = True) -> np.array(float):
    """ Several image normalizations/preprocessing: 
        - downsample number of frames
        - crop to centered image
        - rescale rgb 0-255 value to [-1.0, 1.0] - only if rescale == True

    Returns array of floats
    """

    # downsample and crop the image frames
    arFrames = frames_downsample(arFrames, targetNum)
    arFrames = images_crop(arFrames, height, width)

    if rescale:
        # normalize to [-1.0, 1.0]
        arFrames = images_rescale(arFrames)
    else:
        if np.max(np.abs(arFrames)) > 1.0: warnings.warn("Images not normalized")

    return arFrames

def frames_show(arFrames:np.array, waitTime:int = 100):

    nFrames, _, _, _ = arFrames.shape
    
    for i in range(nFrames):
        cv2.imshow("Frame", arFrames[i, :, :, :])
        cv2.waitKey(waitTime)

    return

def video_length(sVideoPath:str) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", sVideoPath],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    return float(result.stdout)

def videosDir2framesDir(videoDir:str, frameDir:str, framesNorm:int = None, 
    resizeMinDim:int = None, cropShape:tuple = None, classes:int = None):
    """ Extract frames from videos 
    
    Input video structure: ... videoDir / train / class001 / videoname.avi
    Output: ... frameDir / train / class001 / videoname / frames.jpg
    """

    # get videos. Assume videoDir / train / class / video.mp4
    dfVideos = pd.DataFrame(sorted(glob.glob(videoDir + "/*/*/*.*")), columns=["videoPath"])
    print("Located {} videos in {}, extracting to {} ..."\
        .format(len(dfVideos), videoDir, frameDir))
    if len(dfVideos) == 0: raise ValueError("No videos found")

    # restrict to first nLabels
    if classes != None:
        dfVideos.loc[:,"label"] = dfVideos.videoPath.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.label.unique())[:classes]
        dfVideos = dfVideos[dfVideos["label"].isin(liClasses)]
        print("Using only {} videos from first {} classes".format(len(dfVideos), classes))

    counter = 0
    # loop through all videos and extract frames
    for videoPath in dfVideos.videoPath:

        # assemble target diretory
        videoPath = videoPath.replace('\\' , '/') 
        li_videoPath = videoPath.split("/")
        if len(li_videoPath) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_videoPath)))
        sVideoName = li_videoPath[-1].split(".")[0]
        sTargetDir = frameDir + "/" + li_videoPath[-3] + "/" + li_videoPath[-2] + "/" + sVideoName
        
        # check if frames already extracted
        if framesNorm != None and os.path.exists(sTargetDir):
            nFrames = len(glob.glob(sTargetDir + "/*.*"))
            if nFrames == framesNorm: 
                print("Video %5d already extracted to %s" % (counter, sTargetDir))
                counter += 1
                continue
            else: 
                print("Video %5d: Directory with %d instead of %d frames detected" % (counter, nFrames, framesNorm))
        
        # create target directory
        os.makedirs(sTargetDir, exist_ok = True)

        # slice videos into frames
        arFrames = video2frames(videoPath, resizeMinDim)

        # get length and fps
        fVideoSec = video_length(videoPath)
        nFrames = len(arFrames)
        fFPS = nFrames / fVideoSec   

        # downsample
        if framesNorm != None: 
            arFrames = frames_downsample(arFrames, framesNorm)

        # crop images
        if cropShape != None:
            arFrames = images_crop(arFrames, *cropShape)
        
        # write frames to .jpg files
        frames2files(arFrames, sTargetDir)         

        print("Video %5d | %5.1f sec | %d frames | %4.1f fps | saved %s in %s" % (counter, fVideoSec, nFrames, fFPS, str(arFrames.shape), sTargetDir))
        counter += 1      

    return