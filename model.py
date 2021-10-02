import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

class Model():

    def __init__(self, height=None, width=None):
        self.height = height
        self.width = width

    def build_model(self):
        img_rows, img_cols = self.height, self.width
        data_format = keras.backend.image_data_format()

        model = Sequential()
        model.add(Conv2D(64, 3, data_format=data_format,
                        input_shape=(img_rows, img_cols, 1), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size = (2,2), data_format=data_format))
        model.add(Dropout(.5))
        model.add(Conv2D(128, 3, data_format=data_format,
                        padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), data_format=data_format))
        model.add(Dropout(.5))
        model.add(Conv2D(256, 3, data_format=data_format, 
                        padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), data_format=data_format))
        model.add(Dropout(.5))
        model.add(Conv2D(512, 3, data_format=data_format,
                        padding='same', activation='relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(.25))
        model.add(Dense(3))
        model.add(Activation("softmax"))

        return model

