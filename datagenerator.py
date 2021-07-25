""" For neural network training the method Keras.model.fit_generator is used. 
This requires a generator that reads and yields training data to the Keras engine.
"""

import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import Sequence, to_categorical

from preprocess_utils import files2frames, images_normalize

class FramesGenerator(Sequence):
    """ Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, path:str, batchSize:int, nFrames:int, height:int, width:int, nChannels:int, \
        liClassesFull:list = None, shuffle:bool = True):
        """ Assume directory structure: ... / path / class / videoname / frames.jpg
        """

        'Initialization'
        self.batchSize = batchSize
        self.nFrames = nFrames
        self.height = height
        self.width = width
        self.nChannels = nChannels
        self.tuXshape = (nFrames, height, width, nChannels)
        self.shuffle = shuffle

        # retrieve all videos = frame directories
        self.dfVideos = pd.DataFrame(sorted(glob.glob(path + "/*/*")), columns=["sFrameDir"])
        self.nSamples = len(self.dfVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + path)
        print("Detected %d samples in %s ..." % (self.nSamples, path))

        # extract (text) labels from path
        seLabels =  self.dfVideos.sFrameDir.apply(lambda s: s.replace('\\' , '/').split("/")[-2])
        self.dfVideos.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfVideos.sLabel.unique()))

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfVideos.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfVideos.sLabel)
        
        self.on_epoch_end()
        
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.batchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.batchSize:(nStep+1)*self.batchSize]

        # get batch of videos
        dfVideosBatch = self.dfVideos.loc[indexes, :]
        batchSize = len(dfVideosBatch)

        # initialize arrays
        arX = np.empty((batchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((batchSize), dtype = int)

        # Generate data
        for i in range(batchSize):
            # generate data for single video(frames)
            arX[i,], arY[i] = self.__data_generation(dfVideosBatch.iloc[i,:])

        # onehot the labels
        return np.array(arX), np.array(to_categorical(arY, num_classes=self.nClasses))

    def __data_generation(self, seVideo:pd.Series):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        ar_nFrames = files2frames(seVideo.sFrameDir)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        ar_nFrames = ar_nFrames[..., 0:self.nChannels]
        
        ar_fFrames = images_normalize(ar_nFrames, self.nFrames, self.height, self.width, rescale = True)
        
        return ar_fFrames, seVideo.nLabel

    def data_generation(self, seVideo:pd.Series):
        return self.__data_generation(seVideo)

class FeaturesGenerator(Sequence):
    """Reads and yields (preprocessed) I3D features for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, path:str, batchSize:int, tuXshape, \
        liClassesFull:list = None, shuffle:bool = True):
        """
        Assume directory structure:
        ... / path / class / feature.npy
        """

        'Initialization'
        self.batchSize = batchSize
        self.tuXshape = tuXshape
        self.shuffle = shuffle

        # retrieve all feature files
        self.dfSamples = pd.DataFrame(sorted(glob.glob(path + "/*/*.npy")), columns=["path"])
        self.nSamples = len(self.dfSamples)
        if self.nSamples == 0: raise ValueError("Found no feature files in " + path)
        print("Detected %d samples in %s ..." % (self.nSamples, path))

        # test shape of first sample
        arX = np.load(self.dfSamples.path[0])
        if arX.shape != tuXshape: raise ValueError("Wrong feature shape: " + str(arX.shape) + str(tuXshape))

        # extract (text) labels from path
        seLabels =  self.dfSamples.path.apply(lambda s: s.split("/")[-2])
        self.dfSamples.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfSamples.sLabel.unique()))

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfSamples.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfSamples.sLabel)
        
        self.on_epoch_end()

        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.batchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, nStep):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.batchSize:(nStep+1)*self.batchSize]

        # Find selected samples
        dfSamplesBatch = self.dfSamples.loc[indexes, :]
        batchSize = len(dfSamplesBatch)

        # initialize arrays
        arX = np.empty((batchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((batchSize), dtype = int)

        # Generate data
        for i in range(batchSize):
            # generate single sample data
            arX[i,], arY[i] = self.__data_generation(dfSamplesBatch.iloc[i,:])

        # onehot the labels
        return arX, to_categorical(arY, num_classes=self.nClasses)

    def __data_generation(self, seSample:pd.Series):
        'Generates data for 1 sample'
        arX = np.load(seSample.path)

        return arX, seSample.nLabel

class VideoClasses():
    """ Loads the video classes (incl descriptions) from a csv file
    """

    def __init__(self, classFile:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self.dfClass = pd.read_csv(classFile)

        # sort the classes
        self.dfClass = self.dfClass.sort_values("sClass").reset_index(drop=True)
        
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)

        print("Loaded %d classes from %s" % (self.nClasses, classFile))

        return