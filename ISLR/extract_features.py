import os
import numpy as np

from keras.models import load_model

from datagenerator import FramesGenerator

def features_extract_vgg(sFrameBaseDir:str, sFeatureBaseDir:str, nFramesNorm:int = 40):

    vggmodel = load_model('./models/vgg19-flt-64-64.h5')

    # prepare frame generator - without shuffling!
    _, h, w, c = vggmodel.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, 1, nFramesNorm, h, w, c, 
        liClassesFull = None, shuffle=False)

    print("Extract features with %s ... " % vggmodel.name)
    nCount = 0
    # Predict - loop through all samples
    for _, seVideo in genFrames.dfVideos.iterrows():
        
        # ... sFrameBaseDir / class / videoname=frame-directory
        sVideoName = seVideo.sFrameDir.replace('\\' , '/').split("/")[-1]
        sLabel = seVideo.sLabel
        sFeaturePath = sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy"

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already extracted to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue

        # get frames and extract features
        arX, _ = genFrames.data_generation(seVideo)
        arFeatures = vggmodel.predict(arX, verbose=0)

        # save to file
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeaturePath, arFeatures)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeatures.shape), sFeaturePath))
        nCount += 1

    print("%d features saved to files in %s" % (nCount+1, sFeatureBaseDir))
    return