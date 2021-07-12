import cv2
import os
from os.path import join, exists
import argparse
from tqdm import tqdm
import numpy as np

from preprocess_utils import extract_frames, get_grayscale_images, frames_downsample

hc = []

def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not exists(majorData):
        os.makedirs(majorData)

    gesture_folder = os.path.abspath(gesture_folder)

    os.chdir(gesture_folder)
    gestures = os.listdir(os.getcwd())

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    for gesture in tqdm(gestures, unit='actions', ascii=True):
        gesture_path = os.path.join(gesture_folder, gesture)
        os.chdir(gesture_path)

        gesture_frames_path = os.path.join(majorData, gesture)
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]

        for video in tqdm(videos, unit='videos', ascii=True):
            extracted_frames = extract_frames(video)
            downsampled_images = frames_downsample(extracted_frames, 40)
            # grayscale_images = get_grayscale_images(downsampled_images)
            video_path = os.path.join(gesture_frames_path, video)
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            os.chdir(video_path)
            print(os.getcwd())
            for nFrame in range(downsampled_images.shape[0]):
                cv2.imwrite("frame%04d.jpg" % nFrame, downsampled_images[nFrame, ...])
            # name = os.path.abspath(video)
            # cap = cv2.VideoCapture(name)
            # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # count = 0
            # frames = []
            # prvs = None
            # os.chdir(gesture_frames_path)

            # while(True):
            #     ret, frame = cap.read()
            #     if ret == True:
            #         next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #         frames.append(frame)

            #         # FRAME DIFFERENCE
            #         if(prvs is not None):
            #             frame_diff = cv2.absdiff(next, prvs)
            #             ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
            #             # cv2.imshow('frame diff', thres)
            #             # TO-DO --> DO SOME CALCULATION TO CLEAN OUT DIRTY FRAMES

            #             framename = os.path.splitext(video)[0]
            #             framename = framename + "_frame_" + str(count) + ".jpeg"
            #             hc.append([join(gesture_frames_path, framename), gesture, frameCount])

            #             # if not os.path.exists(framename):
            #                 # cv2.imwrite(framename, next)

            #         if cv2.waitKey(1) & 0xFF == ord('q'):
            #             break
            #         count += 1
            #         prvs = next
            #     else:
            #         break

            # downsampled_frames = frames_downsample(np.array(frames), 40)
            # for nFrame in range(downsampled_frames.shape[0]):
            #     cv2.imwrite("frame%04d.jpg" % nFrame, downsampled_frames[nFrame, :, :, :])

            os.chdir(gesture_path)
            # cap.release()
            cv2.destroyAllWindows()

    os.chdir(rootPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Individual Frames from gesture videos.')
    parser.add_argument('gesture_folder', help='Path to folder containing folders of videos of different gestures.')
    parser.add_argument('target_folder', help='Path to folder where extracted frames should be kept.')
    args = parser.parse_args()
    convert(args.gesture_folder, args.target_folder)