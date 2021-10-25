import keras
import numpy as np

from keras import backend as K

def layers_freeze(keModel:keras.Model) -> keras.Model:
    
    print("Freeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = False

    return keModel

def layers_unfreeze(keModel:keras.Model) -> keras.Model:
    
    print("Unfreeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = True

    return keModel

def count_params(keModel:keras.Model):
    for p in keModel.trainable_weights:
        K.count_params(p)

    trainable_count = int(
        np.sum([K.count_params(p) for p in keModel.trainable_weights]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in keModel.non_trainable_weights]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    return