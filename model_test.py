import os
import numpy as np
import pandas as pd
import glob
import csv
import cv2

from keras.models import load_model 

from preprocess_utils import frames_downsample, image_binary, image_grayscale, images_crop, video2frames, images_resize_aspectratio
from optical_flow import frames2flows
from lip_extractor import bodyFrames2LipFrames
from datagenerator import VideoClasses
from predict import probability2label
from model.model_att_lstm import Att_LSTM

def test(params, diVideoSet):

    # model = Att_LSTM()
    # model.load_weights('./saved_models/5-dep-weights.h5')
    model = load_model(params['modelFile'])
    classFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])

    if params["model"] == "lstm":
        vggmodel = load_model('./models/vgg19-flt-64-64.h5')
	
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

        if params['lips']:
            extractedLips = bodyFrames2LipFrames(arFrames)
            arLips = []
            
            for nFrame in range(extractedLips.shape[0]):
                if(params["lip_img"] == "bin" or params["lip_img"] == "gs"):
                    arLip = image_grayscale(extractedLips[nFrame,...])
                    if(params["lip_img"] == "bin"):
                        arLip = image_binary(arLip)
                    arLips.append(arLip)
                else:
                    arLips.append(extractedLips[nFrame,...])
            if(params["lip_img"] == "bin" or params["lip_img"] == "gs"):
                arLips = np.expand_dims(arLips, axis=-1)
            arLips = np.array(arLips)

        if params['oflow']:
            arFlows = images_crop(arFrames, params['inputShape'][0], params['inputShape'][1])
            arFlows = frames2flows(arFlows, bThirdChannel = False, bShow = False)

        # predict video from model			
        print("Predict video with %s ..." % (model.name))
        inputs = []

        if params["model"] == "lstm":
            arFeatures = vggmodel.predict(arLips, verbose=0)
            arXLip = np.expand_dims(arFeatures, axis=0)
            inputs.append(arXLip)

        elif params["model"] == "i3d":
            if params['lips']:
                arXLip = np.expand_dims(arLips, axis=0)
                inputs.append(arXLip)

            if params['oflow']:
                arXFlow = np.expand_dims(arFlows, axis=0)
                inputs.append(arXFlow)

        arProbas = model.predict(inputs, verbose = 1)[0]
        print(arProbas)
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

def test_with_fusion(params, diVideoSet):

    # flow_model = load_model(params['flowModelFile'])
    # lip_model = Att_LSTM()
    # lip_model.load_weights('./saved_models/5-dep-weights.h5')
    # lip_model = load_model(params['lipModelFile'])
    model = load_model(params['fusedModelFile'])
    # baseClassFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nBaseClasses"])
    classFile = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])

    if params["model_lip"] == "lstm":
        vggmodel = load_model('./models/vgg19-flt-64-64.h5')
	
    print("\nStarting gesture recognition live demo ... ")
    print(os.getcwd())
    print(diVideoSet)

    # load label descriptions
    classes = VideoClasses(classFile)
    # baseClasses = VideoClasses(baseClassFile)

    # get videos. Assume videoDir / class / video.mp4
    dfVideos = pd.DataFrame(sorted(glob.glob(params['videoDir'] + "/*/*.*")), columns=["videoPath"])
    print("Located {} videos in {} ..."\
        .format(len(dfVideos), params['videoDir']))
    if len(dfVideos) == 0: raise ValueError("No videos found")

    os.makedirs('./tests', exist_ok=True)
    # create the csv writer
    f = open(params['csv'], 'w', newline='')
    writer = csv.writer(f)    
    writer.writerow(['truth label', 'fused prediction', 'confidence']) 

    # loop through all videos and extract frames
    truth_values = []
    predicted_fused_values = []
    for videoPath in dfVideos.videoPath:

        # assemble target diretory
        videoPath = videoPath.replace('\\' , '/') 
        li_videoPath = videoPath.split("/")
        truthLabel = classes.dfClass.sDetail[classes.dfClass.sClass == li_videoPath[4]].iloc[0]

        # preprocess video
        arFrames = video2frames(videoPath, 360)
        arFrames = frames_downsample(arFrames, diVideoSet['framesNorm'])
        extractedLips = bodyFrames2LipFrames(arFrames)
        arLips = []
            
        for nFrame in range(extractedLips.shape[0]):
            if(params["lip_img"] == "bin" or params["lip_img"] == "gs"):
                arLip = image_grayscale(extractedLips[nFrame,...])
                if(params["lip_img"] == "bin"):
                    arLip = image_binary(arLip)
                arLips.append(arLip)
            else:
                arLips.append(extractedLips[nFrame,...])
        if(params["lip_img"] == "bin" or params["lip_img"] == "gs"):
            arLips = np.expand_dims(arLips, axis=-1)
        arLips = np.array(arLips)

        arTFrames = video2frames(videoPath, 240)
        arTFrames = frames_downsample(arTFrames, diVideoSet['framesNorm'])
        arTFrames = images_resize_aspectratio(arTFrames, diVideoSet["nMinDim"])
        arFlows = images_crop(arTFrames, params['inputShape'][0], params['inputShape'][1])
        arFlows = frames2flows(arFlows, bThirdChannel = False, bShow = False)

        # predict video from model			
        # print("Predict video with %s and %s..." % (flow_model.name, lip_model.name))

        if params["model_lip"] == "lstm":
            arFeatures = vggmodel.predict(arLips, verbose=0)
            arXLip = np.expand_dims(arFeatures, axis=0)

        elif params["model_lip"] == "i3d":
                arXLip = np.expand_dims(arLips, axis=0)
        
        arXFlow = np.expand_dims(arFlows, axis=0)

        # arProbasFlow = flow_model.predict(arXFlow, verbose = 1)[0]
        # classFlow, predictedLabelFlow, fProbaFlow = probability2label(arProbasFlow, classes, nTop = 1)
        # predicted_signs = baseClasses.dfClass.signs[classFlow].split(',')

        # arProbasLip = lip_model.predict(arXLip, verbose = 1)[0]
        # _, predictedLabelLip, fProbaLip = probability2label(arProbasLip, classes, nTop = 1)

        # predicted_signs = np.array(predicted_signs)
        # arProbasLip = np.array(arProbasLip)

        # arProbas = np.ma.array(arProbasLip, mask=predicted_signs)
        # arProbas = arProbas.filled(0.0)
        # arProbas = arProbas / np.sum(arProbas)
        
        arProbas = model.predict([arXFlow, arXLip], verbose = 1)[0]
        # _, predictedLabel, fProba = probability2label(arProbas, classes, nTop = 12)
        # arProbas = arProbasLip + arProbasFlow
        # arProbas = arProbas / np.sum(arProbas)
        _, predictedLabel, fProba = probability2label(arProbas, classes, nTop = 1)

        truth_values.append(truthLabel)
        predicted_fused_values.append(predictedLabel)
        # write a row to the csv file
        writer.writerow([truthLabel, predictedLabel, fProba*100.])

    accuracy = get_accuracy(truth_values, predicted_fused_values)
    print(accuracy)
    # f.close()

    return

if __name__ == '__main__':

    fused_params = {
        "model_lip" : "lstm",
        "model_flow" : "i3d",
        "lip_img" : "rgb",
        # "flowModelFile" : "saved_models/3.h5",
        # "lipModelFile" : "saved_models/20.h5",
        "fusedModelFile" : "saved_models/18-last.h5",
        "videoDir" : "data-set/signs/012/valid",
        "csv" : "./tests/18-last.csv",
        "inputShape" : (240,427,2)
    }

    params = {
        "model" : "lstm",
        "oflow" : False,
        "lips" : True,
        "lip_img" : "rgb",
        "modelFile" : "saved_models/20-last.h5",
        "videoDir" : "data-set/signs/012/valid",
        "csv" : "./tests/20-last.csv",
        "inputShape" : (224,320,2)
    }

    diVideoSet = {
		"sName" : "signs",
        "nBaseClasses" : 5, 	    # number of base classes
		"nClasses" : 12, 	        # number of classes
        "framesNorm" : 40,    		# number of frames per video
        "nMinDim" : 240,            # smaller dimension of extracted video-frames
	}

    # test(params, diVideoSet)
    test_with_fusion(fused_params, diVideoSet)