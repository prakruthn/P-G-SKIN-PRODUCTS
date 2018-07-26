# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as KBack

class LeNet:
    @staticmethod
    def build(width, height, depth, classess):
        #Sequential Model
        model = Sequential()
        ipShape = (height, width, depth)
        
        if(KBack.image_data_format() == "channels_first"):
            ipShape = (depth, height, width)
        
        #first layer
        model.add(Conv2D(20, (5, 5), padding = "same", input_shape = ipShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
        
        #2nd layer
        model.add(Conv2D(50, (5, 5), padding = "same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        model.add(Dense(classess))
        model.add(Activation("softmax"))
        
        return model