import os

from train_lip_model import train_lipImage_end2end
from extract_features import features_extract_vgg

if __name__ == '__main__':

    diVideoSet = {
        "sName" : "signs",
        "nClasses" : 12,        # number of classes
        "framesNorm" : 40,      # number of frames per video
        "nMinDim" : 112,        # smaller dimension of saved video-frames
        "tuShape" : (112, 168), # height, width
    }
    
    # directories
    sFolder         = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["framesNorm"])
    sClassFile      = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    lipDir          = "data-temp/%s/%s/lips"%(diVideoSet["sName"], sFolder)
    lipFeatureDir   = "data-temp/%s/%s/lip-features"%(diVideoSet["sName"], sFolder)

    print("Starting pipeline ...")
    print(os.getcwd())

    features_extract_vgg(lipDir+'/train', lipFeatureDir+'/train')
    features_extract_vgg(lipDir+'/valid', lipFeatureDir+'/valid')

    train_lipImage_end2end(diVideoSet)