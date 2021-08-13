import os
import time
import keras

from datagenerator import VideoClasses, FramesGenerator
from model_i3d import Inception_Inflated3d, add_i3d_top
from train_utils import layers_freeze, layers_unfreeze, count_params

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
    folder          = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    classfile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    oFlowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], folder)
    modelDir        = "model"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 10}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 15}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the dataset classes
    oClasses = VideoClasses(classfile)

    # Load training data
    genFramesTrain = FramesGenerator(oFlowDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 220, 310, 2, oClasses.liClasses)
    genFramesVal = FramesGenerator(oFlowDir + "/valid", nBatchSize, 
        diVideoSet["framesNorm"], 220, 310, 2, oClasses.liClasses)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights='flow_imagenet_and_kinetics',
        input_shape=(diVideoSet["framesNorm"], 220, 310, 2))
    print("Add top layers with %d output classes ..." % oClasses.nClasses)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, oClasses.nClasses, dropout_prob=0.5)

    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-oflow-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    os.makedirs(modelDir, exist_ok=True)
    cpTopBest = keras.callbacks.ModelCheckpoint(filepath = modelDir + "/" + sLog + "-above-best.h5",
        verbose = 1, save_best_only = True)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = modelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr = diTrainTop["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainTop["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpTopBest])
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllBest])

    return

def train_I3D_lipImage_end2end(diVideoSet):
    """ 
    * Loads pretrained I3D model, 
    * reads lip image data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    folder          = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    classfile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    lipsDir        = "data-temp/%s/%s/binary-lips"%(diVideoSet["sName"], folder)
    modelDir        = "model"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 10}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 15}

    nBatchSize = 1

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the dataset classes
    oClasses = VideoClasses(classfile)

    # Load training data
    genFramesTrain = FramesGenerator(lipsDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 112, 168, 1, oClasses.liClasses)
    genFramesVal = FramesGenerator(lipsDir + "/valid", nBatchSize, 
        diVideoSet["framesNorm"], 112, 168, 1, oClasses.liClasses)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights=None,
        input_shape=(diVideoSet["framesNorm"], 112, 168, 1),
        name="lip_")
    print("Add top layers with %d output classes ..." % oClasses.nClasses)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, oClasses.nClasses, dropout_prob=0.5, name="lip_")

    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-lips-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    os.makedirs(modelDir, exist_ok=True)
    cpTopBest = keras.callbacks.ModelCheckpoint(filepath = modelDir + "/" + sLog + "-above-best.h5",
        verbose = 1, save_best_only = True)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = modelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr = diTrainTop["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainTop["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpTopBest])
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllBest])

    return