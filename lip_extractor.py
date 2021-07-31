import numpy as np
import warnings
import tensorflow as tf
import menpo.io as mio
import glob
import os

from PIL import Image as PImage

import detect_face
from preprocess_utils import frames2files, image_resize_aspectratio, image_grayscale, image_binary

def extract_lip_image(minsize:int, threshold:list, factor:float, path:str, \
    pnet:any, rnet:any, onet:any) -> np.array:
    image =  mio.import_image(path)

    # 3 step cascade detection 
    _, points = detect_face.detect_face(image.pixels_with_channels_at_back() * 255, 
                                        minsize, pnet, rnet, onet, threshold, factor)
    I = np.array(PImage.open(path))
    canvas = I.copy()

    if(points.shape[0] == 10):
        distances = (int((points[4, 0] - points[3, 0])/2), np.abs(int((points[9, 0] - points[8, 0])/2)))
        if(points[8, 0] < points[9, 0]):
            center = (int(points[3, 0]) + distances[0], int(points[8, 0]) + distances[1])
        else:
            center = (int(points[3, 0]) + distances[0], int(points[9, 0]) + distances[1])
        top_left = (center[0] - 12, center[1] - 8)
        bottom_right = (center[0] + 12, center[1] + 8)
        canvas = canvas[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
    else: warnings.warn("Too many points obtained for input image")
    
    return np.array(canvas)

def bodyFrames2LipFrames(arFrames:np.array) -> np.array:
    """ Extract lip image sequence from a given video(numpy array of frames)
    """
    minsize = 40
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_detector(sess, './models/face_detection')

    liFrames = []
    for nFrame in range(arFrames.shape[0]):
        # 3 step cascade detection 
        _, points = detect_face.detect_face(arFrames[nFrame, :, :, :], 
                                            minsize, pnet, rnet, onet, threshold, factor)

        if(points.shape[0] == 10):
            distances = (int((points[4, 0] - points[3, 0])/2), np.abs(int((points[9, 0] - points[8, 0])/2)))
            if(points[8, 0] < points[9, 0]):
                center = (int(points[3, 0]) + distances[0], int(points[8, 0]) + distances[1])
            else:
                center = (int(points[3, 0]) + distances[0], int(points[9, 0]) + distances[1])
            top_left = (center[0] - 12, center[1] - 8)
            bottom_right = (center[0] + 12, center[1] + 8)
            frame = arFrames[nFrame, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            frame = image_resize_aspectratio(frame, 112)
            liFrames.append(frame)
        else: warnings.warn("Too many points obtained for input image")
    
    return np.array(liFrames)

def bodyFramesDir2lipFrameDir(bodyBaseDir:str, lipBaseDir:str, minsize:int, \
    threshold:list, factor:float, grayscale:bool=False, binary:bool=False):
    """ Extract Lip frames from body rgb frames (extracted from videos) 
    
    Input videoframe structure: ... bodyBaseDir / train / class001 / videoname / body.jpg
    Output: ... lipBaseDir / train / class001 / videoname / lip.jpg
    """

    # get list of directories with frames: ... / bodyBaseDir/train/class/videodir/body.jpg
    currentDir = os.getcwd()
    os.chdir(bodyBaseDir)
    videos = sorted(glob.glob("*/*/*"))
    os.chdir(currentDir)
    print("Found %d directories=videos with frames in %s" % (len(videos), bodyBaseDir))

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_detector(sess, './models/face_detection')

    # loop over all videos-directories
    counter = 0
    for bodyDir in videos:

        # generate target directory
        lipDir = lipBaseDir + "/" + bodyDir

        if os.path.exists(lipDir):
            print("Video %5d: Lip frames already extracted to %s" % (counter, lipDir))
            counter += 1
            continue

        # retrieve frame files - in ascending order
        # important to sort image files upfront
        path = bodyBaseDir + "/" + bodyDir
        liFiles = sorted(glob.glob(path + "/*.jpg"))
        if len(liFiles) == 0: raise ValueError("No frames found in " + path)

        liFrames = []
        # loop through frames
        print("Video %5d: Extract Lip frames to %s" % (counter, lipDir))
        for frame in liFiles:
            # extract lip frame from body image
            arLipFrame = extract_lip_image(minsize, threshold, factor, frame, pnet, rnet, onet)
            arLipFrame = image_resize_aspectratio(arLipFrame, 112)
            if grayscale:
                arLipFrame = image_grayscale(arLipFrame)
            if binary:
                arLipFrame = image_binary(arLipFrame)
            liFrames.append(arLipFrame)

        frames2files(np.array(liFrames), lipDir)
        counter += 1 

    return