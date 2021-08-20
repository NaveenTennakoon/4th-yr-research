import os
import numpy as np
import pandas as pd
import glob
import csv

from keras.models import load_model 

from preprocess_utils import frames_downsample, images_crop, video2frames
from optical_flow import frames2flows
from lip_extractor import bodyFrames2LipFrames
from datagenerator import VideoClasses
from predict import probability2label

def test(params, diVideoSet):

    model = load_model(params['modelFile'])
    classFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
	
    print("\nStarting gesture recognition live demo ... ")
    print(os.getcwd())
    print(diVideoSet)

    # load label descriptions
    classes = VideoClasses(classFile)

    # get videos. Assume videoDir / class / video.mp4
    dfVideos = pd.DataFrame(sorted(glob.glob(params['videoDir'] + "/*/*.*")), columns=["videoPath"])
    print("Located {} videos in {} ..."\
        .format(len(dfVideos), params['videoDir']))
    if len(dfVideos) == 0: raise ValueError("No videos found")

    os.makedirs('./tests', exist_ok=True)
    # create the csv writer
    f = open(params['csv'], 'w', newline='')
    writer = csv.writer(f)    
    writer.writerow(['truth label', 'predicted label', 'confidence']) 

    # loop through all videos and extract frames
    truth_values = []
    predicted_values = []
    for videoPath in dfVideos.videoPath:

        # assemble target diretory
        videoPath = videoPath.replace('\\' , '/') 
        li_videoPath = videoPath.split("/")
        truthLabel = classes.dfClass.sDetail[classes.dfClass.sClass == li_videoPath[4]].iloc[0]

        # preprocess video
        arFrames = video2frames(videoPath, diVideoSet['nMinDim'])
        arFrames = frames_downsample(arFrames, diVideoSet['framesNorm'])
        if params['lips'] or params['fused']:
            arLips = bodyFrames2LipFrames(arFrames)
        if params['oflow'] or params['fused']:
            arFlows = images_crop(arFrames, params['inputShape'][0], params['inputShape'][1])
            arFlows = frames2flows(arFlows, bThirdChannel = False, bShow = False)

        # predict video from model			
        print("Predict video with %s ..." % (model.name))
        inputs = []
        if params['lips'] or params['fused']:
            arXLip = np.expand_dims(arLips, axis=0)
            inputs.append(arXLip)
        if params['oflow'] or params['fused']:
            arXFlow = np.expand_dims(arFlows, axis=0)
            inputs.append(arXFlow)

        arProbas = model.predict(inputs, verbose = 1)[0]
        _, predictedLabel, fProba = probability2label(arProbas, classes, nTop = 1)

        truth_values.append(truthLabel)
        predicted_values.append(predictedLabel)
        # write a row to the csv file
        writer.writerow([truthLabel, predictedLabel, fProba*100.])

    accuracy = get_accuracy(truth_values, predicted_values)
    print(accuracy)
    f.close()

    return

def get_accuracy(truth_values, predicted_values):
    """ Calculate the accuracy of classification based on predicted results

    truth_values(np.array)      - actual labels of the data,
    predicted_values(np.array)  - predicted labels of the data
    """

    correct_predictions = 0
    total_len = len(truth_values)
    for i in range(total_len):
        if(truth_values[i] == predicted_values[i]):
            correct_predictions = correct_predictions + 1
    accuracy = (correct_predictions/total_len)*100

    return accuracy

if __name__ == '__main__':

    params = {
        "fused" : False,
        "oflow" : True,
        "lips" : False,
        "modelFile" : "model/4.h5",
        "videoDir" : "data-set/signs/012/valid",
        "csv" : "./tests/4.csv",
        "inputShape" : (240,427,2)
    }

    diVideoSet = {
		"sName" : "signs",
		"nClasses" : 5, 	        # number of classes
        "framesNorm" : 40,    		# number of frames per video
		"nMinDim" : 240,        	# smaller dimension of extracted video-frames
	}

    test(params, diVideoSet)