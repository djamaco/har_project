import tensorflow as tf

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

def create_lrcn_bleed_model(input_shape, classes_count):
    """
    Source: https://bleedaiacademy.com/human-activity-recognition-using-tensorflow-cnn-lstm/
    Creates a Long-term Recurrent Convolutional Network (LRCN) model for human activity recognition.

    Args:
        input_shape (tuple): The shape of the input data.
        classes_count (int): The number of classes to predict.

    Returns:
        tf.keras.models.Sequential: The LRCN model.
    """
    model = Sequential()
    
    # First layer: TimeDistributed Conv2D with 16 filters, ReLU activation
    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), padding='same', activation='relu'), input_shape=input_shape))
    # First TimeDistributed MaxPooling to reduce dimensionality
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))

    # Dropout layer to reduce overfitting by dropping out 25% of the neurons
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))
    
    # Second layer: TimeDistributed Conv2D with 32 filters, ReLU activation
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), padding='same', activation='relu')))
    # Second TimeDistributed MaxPooling layer
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    # Another Dropout layer
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))
    
    # Third layer: TimeDistributed Conv2D with 64 filters, ReLU activation
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu')))
    # Third TimeDistributed MaxPooling layer
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    # Another Dropout layer
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))
    
    # Fourth layer: TimeDistributed Conv2D with 64 filters, ReLU activation
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', activation='relu')))
    # Fourth TimeDistributed MaxPooling layer
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
                                      
    # Flatten the output to prepare it for the LSTM layer
    model.add(layers.TimeDistributed(layers.Flatten()))
                                      
    # LSTM layer with 32 units
    model.add(layers.LSTM(32))
                                      
    # Dense layer with softmax activation for classification
    model.add(layers.Dense(classes_count, activation='softmax'))
    
    return model

def create_lrcn_djamaco_model(input_shape, classes_count):
    """
    Creates a Long-term Recurrent Convolutional Network (LRCN) model for human activity recognition.

    Args:
        classes_count (int): The number of classes for classification.

    Returns:
        model (Sequential): The LRCN model.

    """
    model = Sequential()
    MAGIC_NUMBER = 16
    
    # First layer: TimeDistributed wrapper around a Conv2D layer with 16 filters
    model.add(layers.TimeDistributed(layers.Conv2D(MAGIC_NUMBER, (3, 3),  padding='same'), input_shape=input_shape))
    # Activation layer with ReLU (Rectified Linear Unit)
    model.add(layers.Activation('relu'))
    # TimeDistributed MaxPooling to reduce dimensionality
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    # Dropout layer to reduce overfitting by dropping out 25% of the neurons
    model.add(layers.Dropout(0.25))

    # Second layer: TimeDistributed Conv2D with 32 filters
    model.add(layers.TimeDistributed(layers.Conv2D(MAGIC_NUMBER * 2, (3, 3), padding='same', activation='relu')))
    # Another ReLU activation layer
    model.add(layers.Activation('relu'))
    # Second TimeDistributed MaxPooling layer
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    # Another Dropout layer
    model.add(layers.Dropout(0.25))

    # Third layer: TimeDistributed Conv2D with 64 filters
    model.add(layers.TimeDistributed(layers.Conv2D(MAGIC_NUMBER * 4, (3, 3), padding='same', activation='relu')))
    # ReLU activation layer
    model.add(layers.Activation('relu'))
    # Third TimeDistributed MaxPooling layer
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    # Flatten the output to prepare it for the LSTM layer
    model.add(layers.TimeDistributed(layers.Flatten()))

    # LSTM layer with 64 units
    model.add(layers.LSTM(MAGIC_NUMBER * 4, return_sequences=False))

    # Dense layer with softmax activation for classification
    model.add(layers.Dense(classes_count, activation='softmax'))

    return model