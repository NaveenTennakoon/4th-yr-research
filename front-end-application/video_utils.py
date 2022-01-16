""" Utilites to process video inputs.
"""

import numpy as np
import cv2
import tensorflow as tf

from detect_face import create_detector, detect_face

def get_byte_image(frame):
    _, frame = cv2.imencode('.jpg', frame)
    
    return frame.tobytes()

def load_lip_detector():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = create_detector(sess, './face_detection')
    return pnet, rnet, onet

def bodyFrames2LipFrames(arFrames, pnet, rnet, onet):
    """ Extract lip image sequence from a given video(numpy array of frames)
    """
    minsize = 40
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709

    liFrames = []
    for nFrame in range(arFrames.shape[0]):
        # 3 step cascade detection 
        _, points = detect_face(arFrames[nFrame, :, :, :], 
                                            minsize, pnet, rnet, onet, threshold, factor)

        try:
            distances = (int((points[4, 0] - points[3, 0])/2), np.abs(int((points[9, 0] - points[8, 0])/2)))
            if(points[8, 0] < points[9, 0]):
                center = (int(points[3, 0]) + distances[0], int(points[8, 0]) + distances[1])
            else:
                center = (int(points[3, 0]) + distances[0], int(points[9, 0]) + distances[1])
            top_left = (center[0] - 8, center[1] - 8)
            bottom_right = (center[0] + 8, center[1] + 8)
            frame = arFrames[nFrame, top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            frame = cv2.cvtColor(frame, 3)
            frame = cv2.resize(frame, dsize = (0,0), fx = 4, fy = 4, interpolation=cv2.INTER_LINEAR)
            liFrames.append(frame)
        except:
            print("Error during mouth segmentation")
            return None
    
    return np.array(liFrames)