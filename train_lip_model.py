import os
import keras
import time

from model.model_att_lstm import Att_LSTM
from datagenerator import VideoClasses, FeaturesGenerator
from train_utils import count_params

def train_lipImage_end2end(diVideoSet):
    """ 
    * Loads pretrained I3D model, 
    * reads lip image data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    # directories
    classfile      = "data/slsl-22/annotations/gloss_class.csv"    
    lipFeatureDir = "data/slsl-22/islsl-22/lip-features"
    modelDir        = "saved_models"

    diTrain = {
        "fLearn" : 1e-4,
        "nEpochs" : 200}

    nBatchSize = 2

    print("\nStarting Attention-LSTM end2end training ...")
    print(os.getcwd())

    # read the dataset classes
    oClasses = VideoClasses(classfile)

    # Load training data
    genFeaturesTrain = FeaturesGenerator(lipFeatureDir + "/train", nBatchSize, 
        (diVideoSet["framesNorm"], 2048), oClasses.liClasses)
    genFeaturesVal = FeaturesGenerator(lipFeatureDir + "/test", nBatchSize, 
        (diVideoSet["framesNorm"], 2048), oClasses.liClasses)

    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-2-dep"
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    os.makedirs(modelDir, exist_ok=True)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = modelDir + "/" + sLog + ".h5", 
        save_best_only = True)
    cpAllLast = keras.callbacks.ModelCheckpoint(filepath = modelDir + "/" + sLog + "-last.h5")
        
    model = Att_LSTM()
    
    # Fit entire model
    print("Train all layers with generator: %s" % (diTrain))
    optimizer = keras.optimizers.Adam(lr = diTrain["fLearn"])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(model)    

    model.fit_generator(
        genFeaturesTrain,
        validation_data = genFeaturesVal,
        epochs = diTrain["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return