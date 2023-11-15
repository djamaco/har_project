import tensorflow as tf
from config import FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

def create_lrcn_model(classes_count):
    model = Sequential()
    
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4)))) 
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))
    
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))
    
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))
    
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))
                                      
    model.add(layers.TimeDistributed(layers.Flatten()))
                                      
    model.add(layers.LSTM(32))
                                      
    model.add(layers.Dense(classes_count, activation = 'softmax'))
    
    return model