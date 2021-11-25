import os
import glob
import pandas as pd

from preprocess_utils import video2frames, frames2files, sVideosDir2wFramesDir
from optical_flow import framesDir2flowsDir

if __name__ == '__main__':

    framesDir2flowsDir(
        '../../../Coding/4th-yr-research/ISLR/data/slsl-22/islsl-22',
        '../../../Coding/4th-yr-research/ISLR/data/slsl-22/optflow',
        framesNorm = 20
    )