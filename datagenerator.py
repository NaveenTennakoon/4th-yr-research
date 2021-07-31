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

class MultipleInputGenerator(Sequence):
    """ Read and yields video frames/optical flow and lip frames for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    This generator is used for joint fusion training
    """

    def __init__(self, flowPath:str, lipPath:str, batchSize:int, nFrames:int, flowHeight:int, flowWidth:int, \
        lipHeight:int, lipWidth:int, nFlowChannels:int, nLipChannels:int, liClassesFull:list = None, \
        shuffle:bool = True):
        """ Assume directory structure: ... / flowPath / class / videoname / frames.jpg
                                      : ... / lipPath / class / videoname / frames.jpg
        """

        'Initialization'
        self.batchSize = batchSize
        self.nFrames = nFrames
        self.flowHeight = flowHeight
        self.flowWidth = flowWidth
        self.nFlowChannels = nFlowChannels
        self.lipHeight = lipHeight
        self.lipWidth = lipWidth
        self.nLipChannels = nLipChannels
        self.tuFlowXShape = (nFrames, flowHeight, flowWidth, nFlowChannels)
        self.tuLipXShape = (nFrames, lipHeight, lipWidth, nLipChannels)
        self.shuffle = shuffle

        # retrieve all videos = frame directories
        self.flowVideos = pd.DataFrame(sorted(glob.glob(flowPath + "/*/*")), columns=["sFrameDir"])
        self.nSamples = len(self.flowVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + flowPath)
        print("Detected %d samples in %s ..." % (self.nSamples, flowPath))

        self.lipVideos = pd.DataFrame(sorted(glob.glob(lipPath + "/*/*")), columns=["sFrameDir"])
        if self.nSamples != len(self.lipVideos): raise ValueError("Directories have different number of samples")
        self.nSamples = len(self.lipVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + lipPath)
        print("Detected %d samples in %s ..." % (self.nSamples, lipPath))

        # extract (text) labels from flowPath
        seLabels =  self.flowVideos.sFrameDir.apply(lambda s: s.replace('\\' , '/').split("/")[-2])
        self.flowVideos.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.flowVideos.sLabel.unique()))

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
        self.flowVideos.loc[:, "nLabel"] = trLabelEncoder.transform(self.flowVideos.sLabel)
        
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
        flowVideosBatch = self.flowVideos.loc[indexes, :]
        lipVideosBatch = self.lipVideos.loc[indexes, :]
        batchSize = len(flowVideosBatch)

        # initialize arrays
        arFlowX = np.empty((batchSize, ) + self.tuFlowXShape, dtype = float)
        arLipX = np.empty((batchSize, ) + self.tuLipXShape, dtype = float)
        arY = np.empty((batchSize), dtype = int)

        # Generate data
        for i in range(batchSize):
            # generate data for single video(frames)
            arFlowX[i,], arLipX[i,], arY[i] = self.__data_generation(flowVideosBatch.iloc[i,:], lipVideosBatch.iloc[i,:])

        # onehot the labels
        return [np.array(arFlowX), np.array(arLipX)], np.array(to_categorical(arY, num_classes=self.nClasses))

    def __data_generation(self, seFlowVideo:pd.Series, seLipVideo:pd.Series):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        arFlowFrames = files2frames(seFlowVideo.sFrameDir)
        arLipFrames = files2frames(seLipVideo.sFrameDir)

        # only use the first nFlowChannels (typically 3, but maybe 2 for optical flow)
        arFlowFrames = arFlowFrames[..., 0:self.nFlowChannels]
        arLipFrames = arLipFrames[..., 0:self.nLipChannels]
        
        ar_fFrames = images_normalize(arFlowFrames, self.nFrames, self.flowHeight, self.flowWidth, rescale = True)
        ar_lFrames = images_normalize(arLipFrames, self.nFrames, self.lipHeight, self.lipWidth, rescale = True)
        
        return ar_fFrames, ar_lFrames, seFlowVideo.nLabel

    def data_generation(self, seFlowVideo:pd.Series, seLipVideo:pd.Series):
        return self.__data_generation(seFlowVideo, seLipVideo)

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