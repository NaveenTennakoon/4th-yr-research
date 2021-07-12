import cv2
import os
from os.path import exists
import argparse
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def convert(gesture_folder, target_folder, fps, target_width, target_height):
    root_path = os.getcwd()
    target_path = os.path.abspath(target_folder)

    if not exists(target_path):
        os.makedirs(target_path)

    gesture_folder = os.path.abspath(gesture_folder)
    os.chdir(gesture_folder)
    gestures = os.listdir(os.getcwd())

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory for resized videos: %s\n" % (target_path))

    for category in tqdm(gestures, unit='actions', ascii=True):
        gesture_path = os.path.join(gesture_folder, category)
        os.chdir(gesture_path)

        resized_videos_path = os.path.join(target_path, category)
        if not os.path.exists(resized_videos_path):
            os.makedirs(resized_videos_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]

        for video in tqdm(videos, unit='videos', ascii=True):
            name = os.path.abspath(video)
            cap = cv2.VideoCapture(name)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            videoname = os.path.splitext(video)[0] + ".mp4"
            os.chdir(resized_videos_path)
            if not os.path.exists(videoname):
                out = cv2.VideoWriter(videoname, fourcc, fps, (target_width, target_height))

                while(True):
                    ret, frame = cap.read()
                    if(ret != True):
                        break
                    frame = cv2.resize(frame, (target_width, target_height), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
                    out.write(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    os.chdir(root_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize Videos.')
    parser.add_argument('gesture_folder', help='Path to folder containing folders of videos of different gestures.', type=str)
    parser.add_argument('target_folder', help='Path to folder where extracted resized videos should be kept.', type=str)
    parser.add_argument('fps', help='The FPS count in converted video.', default=30, type=int)
    parser.add_argument('target_width', help='The width of resized videos.', default=480, type=int)
    parser.add_argument('target_height', help='The height of resized videos.',default=640, type=int)
    args = parser.parse_args()
    print(args)
    convert(args.gesture_folder, args.target_folder, args.fps, args.target_width, args.target_height)