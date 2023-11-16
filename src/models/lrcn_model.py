import tensorflow as tf

from src.config import FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

def create_lrcn_model_bleed(classes_count):
    """
    Source: https://bleedaiacademy.com/human-activity-recognition-using-tensorflow-cnn-lstm/
    Creates a Long-term Recurrent Convolutional Network (LRCN) model for human activity recognition.

    Args:
        classes_count (int): The number of classes to predict.

    Returns:
        tf.keras.models.Sequential: The LRCN model.
    """
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
                                      
    model.add(layers.TimeDistributed(layers.Flatten()))
                                      
    model.add(layers.LSTM(32))
                                      
    model.add(layers.Dense(classes_count, activation = 'softmax'))
    
    return model

def create_lrcn_model(classes_count):
    model = Sequential()
    FCN_MAGIC_NUMBER = 16
    
    # TimeDistributed CNN layers
    model.add(layers.TimeDistributed(layers.Conv2D(FCN_MAGIC_NUMBER, (3, 3),  padding='same'),
                                     input_shape=(FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Conv2D(FCN_MAGIC_NUMBER * 2, (3, 3), padding='same', activation='relu')))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Conv2D(FCN_MAGIC_NUMBER * 4, (3, 3), padding='same', activation='relu')))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Flatten()))

    # LSTM layer
    model.add(layers.LSTM(FCN_MAGIC_NUMBER * 4, return_sequences=False))

    # Dense layer
    model.add(layers.Dense(classes_count, activation='softmax'))

    return model