import cv2
import os
import numpy as np

def extract_frames(video:str) -> np.array:
    """ Extract the frameset from a given video path
    """

    name = os.path.abspath(video)
    cap = cv2.VideoCapture(name)
    return_frames = []

    while(True):
        ret, frame = cap.read()
        if ret == False:
            break
        return_frames.append(frame)
    
    return np.array(return_frames)

def get_grayscale_images(frames:np.array) -> np.array:
    """ Returns grayscale images for provided set of frames as numpy arrays
    """

    return_frames = []

    for frame_num in range(frames.shape[0]):
        gray_image = get_grayscale_image(frames[frame_num, ...])
        return_frames.append(gray_image)

    return np.array(return_frames)

def get_grayscale_image(frame:np.array) -> np.array:
    """ Returns a grayscale image of the provided frame
    """  

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return gray_image

def fdc_keyframes(frames:np.array, thres:int) -> np.array:
    """ Extract keyframes by calculating absolute frame difference
    through frame iteration. 'thres' provides the threshold value.
    If the frame difference is below the threshold, the current frame
    is ignored and the next frame is taken
    """

    return_frames = []

    for frame_num in range(frames.shape[0] - 1):
        if(frame_num == 0):
            prvs = frames[frame_num, ...]
            return_frames.append(prvs)
        next = frames[frame_num+1, ...]
        diff = cv2.absdiff(next, prvs)
        non_zero_count = np.count_nonzero(diff)
        if non_zero_count > thres:
            return_frames.append(next)
            prvs = next

    return np.array(return_frames)

def frames_downsample(frames:np.array, nFramesTarget:int) -> np.array:
    """ Adjust number of frames (eg 123) to nFramesTarget (eg 79)
    works also if originally less frames than nFramesTarget
    """

    nSamples, _, _, _ = frames.shape
    if nSamples == nFramesTarget: return frames

    fraction = nSamples / nFramesTarget
    index = [int(fraction * i) for i in range(nFramesTarget)]
    liTarget = [frames[i, ...] for i in index]

    return np.array(liTarget)

def calculate_optflow(frames:np.array, algorithm:str='dtvl1') -> np.array:
    hsv = np.zeros_like(frames[0, ...])
    hsv[...,1] = 255
    return_frames = []

    for frame_num in range(frames.shape[0] - 1):
        # Farneback algorithm
        # flow = cv2.calcOpticalFlowFarneback(frames[frame_num, :, :, :], frames[frame_num+1, :, :, :], None, 0.5, 3, 5, 3, 5, 1.2, 0)

        # tv-l1 dual optical flow
        # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(scaleStep = 0.5, warps = 3, epsilon = 0.02)
        
        # tv-l1 dual optical flow - very fast
        # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(theta = 0.1, nscales = 1, scaleStep = 0.5, warps = 1, epsilon = 0.1)
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = optical_flow.calc(frames[frame_num, :, :, :], frames[frame_num+1, :, :, :], None)
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return_frames.append(bgr)

    return np.array(return_frames)

def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
    """
    This function returns 1 for the frames in which the area
    after subtraction with previous frame is greater than minimum area
    defined.
    Thus expensive computation of human detection face detection
    and face recognition is not done on all the frames.
    Only the frames undergoing significant amount of change (which is controlled min_area)
    are processed for detection and recognition.
    """
    frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp = 0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > min_area:
            temp = 1
    return temp 

def motion_detector(self, img):
    occupied = False
    # resize the frame, convert it to grayscale, and blur it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    
    if self.avg is None:
        print("[INFO] starting background model...")
        self.avg = gray.copy().astype("float")
    
    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, self.avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))
    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, 5, 255,
        cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 5000:
            pass
        occupied = True
    
    return occupied 

def extract_keyframes():
    video_path = "/Users/anmoluppal/Downloads/SampleVideo_1280x720_1mb.mp4"
    p_frame_thresh = 300000 # You may need to adjust this threshold

    cap = cv2.VideoCapture(video_path)
    # Read the first frame.
    ret, prev_frame = cap.read()

    while ret:
        ret, curr_frame = cap.read()

        if ret:
            diff = cv2.absdiff(curr_frame, prev_frame)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count > p_frame_thresh:
                print("Got P-Frame")
            prev_frame = curr_frame