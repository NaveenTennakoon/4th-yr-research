import os

from train_lip_model import train_lipImage_end2end
from extract_features import features_extract_vgg

if __name__ == '__main__':

    diVideoSet = {
        "sName" : "slsl-22",
        "nClasses" : 12,       # number of classes
        "framesNorm" : 40,     # number of frames per video
        "nMinDim" : 96,        # smaller dimension of saved video-frames
        "tuShape" : (96, 128), # height, width
    }

    # directories
    sClassFile      = "data/slsl-22/annotations/gloss_class.csv"    
    lipDir          = "data/slsl-22/islsl-22/lips"
    lipFeatureDir = "data/slsl-22/islsl-22/lip-features"

    print("Starting pipeline ...")
    print(os.getcwd())

    features_extract_vgg(lipDir+'/train', lipFeatureDir+'/train')
    features_extract_vgg(lipDir+'/test', lipFeatureDir+'/test')

    train_lipImage_end2end(diVideoSet)