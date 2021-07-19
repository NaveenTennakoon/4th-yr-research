import keras

keModel = keras.models.load_model('model/20210717-0528-chalearn020-oflow-i3d-above-best.h5')
keModel.summary()