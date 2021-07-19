import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import keras
from keras import backend as K

from datagenerator import VideoClasses, FramesGenerator

def layers_unfreeze_specific(keModel:keras.Model, name:str) -> keras.Model:
    
    print("Unfreeze %s layer in Model %s" % (name, keModel.name))
    for layer in keModel.layers:
        if layer.name == name:
            layer.trainable = True
        else: layer.trainable = False

    return keModel

def layers_unfreeze(keModel:keras.Model) -> keras.Model:
    
    print("Unfreeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = True

    return keModel


def count_params(keModel:keras.Model):
    for p in keModel.trainable_weights:
        K.count_params(p)

    trainable_count = int(
        np.sum([K.count_params(p) for p in keModel.trainable_weights]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in keModel.non_trainable_weights]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    return



def train_I3D_oflow_end2end(diVideoSet):
    """ 
    * Loads pretrained I3D model, 
    * reads optical flow data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model"

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the ChaLearn classes
    oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = FramesGenerator(sOflowDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 224, 224, 2, oClasses.liClasses)
    genFramesVal = FramesGenerator(sOflowDir + "/valid", nBatchSize, 
        diVideoSet["framesNorm"], 224, 224, 2, oClasses.liClasses)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = keras.models.load_model('model/20210717-0528-chalearn020-oflow-i3d-above-best.h5')
    print("Add top layers with %d output classes ..." % oClasses.nClasses)
        
    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-oflow-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    cpAllLast = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-last.h5", verbose = 0)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)
    
    # Fit entire I3D model
    print("Finetune specific I3D layers with generator: %s" % (diTrainAll))
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(learning_rate = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return
    
    
if __name__ == '__main__':

    diVideoSet = {"sName" : "signs",
        "nClasses" : 12,   # number of classes
        "framesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (720, 1280), # height, width
        "nFpsAvg" : 30,
        "nFramesAvg" : 90, 
        "fDurationAvG" : 3.0} # seconds 
    
    train_I3D_oflow_end2end(diVideoSet)