"""
https://github.com/FrederikSchorr/sign-language

Utilities for predicting a output label with a neural network 
"""

import numpy as np

from datagenerator import VideoClasses

def probability2label(arProbas:np.array, oClasses:VideoClasses, nTop:int = 3):
    """ Return
        3-tuple: predicted nLabel, sLabel, fProbability
        in addition print nTop most probable labels
    """

    arTopLabels = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[arTopLabels]

    for i in range(nTop):
        sClass = oClasses.dfClass.sClass[arTopLabels[i]] + " " + oClasses.dfClass.sDetail[arTopLabels[i]]
        print("Top %d: [%3d] %s (confidence %.1f%%)" % \
            (i+1, arTopLabels[i], sClass, arTopProbas[i]*100.))
        
    return arTopLabels[0], oClasses.dfClass.sDetail[arTopLabels[0]], arTopProbas[0]