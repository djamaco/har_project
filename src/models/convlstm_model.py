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
    
    # First ConvLSTM layer with 4 filters, tanh activation function, and recurrent dropout of 0.2.
    model.add(layers.ConvLSTM2D(filters = 4, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True, input_shape = input_shape))
    # Max pooling layer to reduce the spatial dimensions.
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    
    # Dropout layer to prevent overfitting.
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    # Second ConvLSTM layer with 8 filters.
    model.add(layers.ConvLSTM2D(filters = 8, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    # Max pooling layer following the second ConvLSTM layer.
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    # Another dropout layer.
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    # Third ConvLSTM layer with 14 filters.
    model.add(layers.ConvLSTM2D(filters = 14, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    # Max pooling layer following the third ConvLSTM layer.
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    
    # Additional dropout layer.
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))
    
    # Fourth and final ConvLSTM layer with 16 filters.
    model.add(layers.ConvLSTM2D(filters = 16, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                         recurrent_dropout=0.2, return_sequences=True))
    # Final max pooling layer.
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))    
    
    # Flatten the output to feed it into a dense layer.
    model.add(layers.Flatten())
    # Dense layer with a softmax activation for classification.
    model.add(layers.Dense(classes_count, activation = "softmax"))

    return model