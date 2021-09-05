import csv
import pandas as pd
import glob

from preprocess_utils import video_length, video2frames

def annotate(videoDir:str, csv_file:str):
    """ Write information about videos to annotation csv file 
    
    Input video structure: ... videoDir / train / class001 / videoname.avi
    """

    # get videos. Assume videoDir / train / class / video.mp4
    dfVideos = pd.DataFrame(sorted(glob.glob(videoDir + "/*/*/*.*")), columns=["videoPath"])
    print("Located {} videos in {} ...".format(len(dfVideos), videoDir))
    if len(dfVideos) == 0: raise ValueError("No videos found")

    f = open(csv_file, 'a+', newline='')
    writer = csv.writer(f)    

    counter = 0
    # loop through all videos and write to csv
    for videoPath in dfVideos.videoPath:

        # assemble target diretory
        videoPath = videoPath.replace('\\' , '/') 
        li_videoPath = videoPath.split("/")
        if len(li_videoPath) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_videoPath)))
        sClass = li_videoPath[-2]
        sVideoName = li_videoPath[-1].split(".")[0]

        # slice videos into frames
        arFrames = video2frames(videoPath, None)
        
        # get length and fps
        fVideoSec = video_length(videoPath)
        nFrames = len(arFrames)
        fFPS = nFrames / fVideoSec

        writer.writerow([counter, sClass, sVideoName, '001', fFPS, fVideoSec, nFrames, str(arFrames.shape[1:]), ''])
        counter += 1      

    return

if __name__ == '__main__':

    videoDir = "data-set/slsl-22/022"
    csv_file = "data-set/slsl-22/022/annotations.csv"

    annotate(videoDir, csv_file)