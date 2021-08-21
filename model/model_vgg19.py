from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model
from keras import backend as K

from train_utils import layers_freeze

# Load pretrained vgg19 model and remove top layer 
if K.image_data_format() == 'channels_first':
    input_shape = (3, 128, 160)
else:
    input_shape = (128, 160, 3)
vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

input = vgg.input
output = vgg.output
flatten = Flatten()(output)
fc = Dense(units=4096, activation='relu', name='fc')(flatten)

#create new model
vggmodel = Model(input, fc, name='vgg19-mod')
vggmodel = layers_freeze(vggmodel)

vggmodel.save('../models/vgg19-mod.h5')