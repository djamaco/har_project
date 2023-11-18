import tensorflow as tf

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

def create_convlstm_bleed_model(input_shape, classes_count):
    """
    Source: https://bleedaiacademy.com/human-activity-recognition-using-tensorflow-cnn-lstm/
    Creates a ConvLSTM model for human activity recognition.

    Args:
    classes_count (int): Number of classes for classification.

    Returns:
    model (keras.Sequential): A ConvLSTM model for human activity recognition.
    """
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = input_shape))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    model.add(layers.ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    model.add(layers.ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))    
    
    model.add(layers.Flatten())
    model.add(layers.Dense(classes_count, activation = "softmax"))

    return model