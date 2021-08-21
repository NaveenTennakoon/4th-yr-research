import os
import time
import keras

from datagenerator import VideoClasses, MultipleInputGenerator
from model_i3d import late_fused_model, early_fused_model
from train_i3d import layers_unfreeze, count_params

def train_i3d_oflow_lip_late_fusion(diVideoSet):
    """ 
    * Loads pretrained I3D lip and optflow models, 
    * Creates a model with late fusion after final softmax layers
    * Fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder          = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    sLipDir          = "data-temp/%s/%s/lips"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model"

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 1

    print("\nStarting I3D end2end training with late fusion ...")
    print(os.getcwd())

    # read the classes
    oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = MultipleInputGenerator(sOflowDir + "/train", sLipDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 224, 320, 128, 160, 2, 3, oClasses.liClasses)
    genFramesVal = MultipleInputGenerator(sOflowDir + "/train", sLipDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 224, 320, 124, 160, 2, 3, oClasses.liClasses)

    # Load pretrained i3d models 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = keras.models.load_model('model/12-oflow-tl-rc-top-best.h5')
    print("Load pretrained I3D Lip model ...")
    keI3DLip = keras.models.load_model('model/12-lip-tl-top-best.h5')
    model = late_fused_model(keI3DOflow, keI3DLip)
        
    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-fused-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)
    
    # Fit entire model
    print("Finetune all layers with generator: %s" % (diTrainAll))
    model = layers_unfreeze(model)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(model)

    model.fit_generator(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllBest])

    return

def train_i3d_oflow_lip_early_fusion(diVideoSet):
    """ 
    * Loads pretrained I3D lip and optflow models, 
    * Creates a model with early fusion after concatenation layers
    * Fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder          = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    sLipDir          = "data-temp/%s/%s/lips"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model"

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 1

    print("\nStarting I3D end2end training with early fusion ...")
    print(os.getcwd())

    # read the classes
    oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = MultipleInputGenerator(sOflowDir + "/train", sLipDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 224, 320, 128, 160, 2, 3, oClasses.liClasses)
    genFramesVal = MultipleInputGenerator(sOflowDir + "/train", sLipDir + "/train", nBatchSize, 
        diVideoSet["framesNorm"], 224, 320, 128, 160, 2, 3, oClasses.liClasses)

    # Load pretrained i3d models 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = keras.models.load_model('model/12-oflow-tl-rc-top-best.h5')
    print("Load pretrained I3D Lip model ...")
    keI3DLip = keras.models.load_model('model/12-lip-tl-top-best.h5')
    model = early_fused_model(keI3DOflow, keI3DLip)
        
    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-early-fused-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    model = layers_unfreeze(model)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(model)

    model.fit_generator(
        genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllBest])

    return
    
if __name__ == '__main__':

    late_fusion = True
    early_fusion = False

    diVideoSet = {
        "sName" : "signs",
        "nClasses" : 12,        # number of classes
        "framesNorm" : 40,      # number of frames per video
        "nMinDim" : 240,        # smaller dimension of saved video-frames
        "tuShape" : (720, 1280),# height, width
    }
    
    if late_fusion:
        train_i3d_oflow_lip_late_fusion(diVideoSet)
    if early_fusion:
        train_i3d_oflow_lip_early_fusion(diVideoSet)