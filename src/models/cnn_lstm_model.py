import tensorflow as tf

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model
K = tf.keras.backend


def create_cnn_lstm_sylvia_model(input_shape, classes_count):
    """
    Source: https://www.kaggle.com/code/sylvianclee/human-activity-recognition-cnn-lstm
    Creates a CNN-LSTM model. 

    Args:
        classes_count (int): The number of classes in the dataset.

    Returns:
        keras.models.Sequential: The CNN-LSTM model.
    """
    model = Sequential()
    model.add(layers.Reshape((input_shape[0], input_shape[1] * input_shape[2] , input_shape[3]), input_shape=input_shape))
    
    model.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))

    model.add(layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=3, activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
    model.add(layers.TimeDistributed(layers.Dropout(0.5)))

    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(100))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(classes_count, activation='softmax'))

    return model

def get_attention_based(inputs, time_steps):
    # Permute the dimensions of the input to make time_steps the last dimension
    a = layers.Permute((2, 1))(inputs)
    # Dense layer to learn the attention weights with softmax activation
    a = layers.Dense(time_steps, activation='softmax')(a)
    # Permute back the dimensions to their original configuration
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    # Multiply the inputs with the attention probabilities
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    # Sum the attended features over the time steps
    output_summed = layers.Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
    return output_summed

def create_3dcnn_lstm_attention_djamaco_model(input_shape, classes_count):
    # Define the input layer with the given input shape
    input_layer = layers.Input(input_shape)

    MAGIC_NUMBER = 16

    # First 3D Convolution layer with 16 filters
    x = layers.Conv3D(MAGIC_NUMBER, (3, 3, 3), activation='relu', padding='same')(input_layer)
    # Max Pooling to reduce spatial dimensions
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    # Batch normalization to stabilize learning
    x = layers.BatchNormalization()(x)

    # Second 3D Convolution layer with 32 filters
    x = layers.Conv3D(MAGIC_NUMBER * 2, (3, 3, 3), activation='relu', padding='same')(x)
    # Max Pooling following the second convolution layer
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    # Batch normalization
    x = layers.BatchNormalization()(x)

    # Flatten the output to prepare it for LSTM
    x = layers.TimeDistributed(layers.Flatten())(x)

    # LSTM layer with 64 units, returning sequences to connect to the attention layer
    x = layers.LSTM(MAGIC_NUMBER * 4, return_sequences=True)(x)

    # Apply attention mechanism
    x = get_attention_based(x, input_shape[0])

    # Flatten the output of the attention layer
    x = layers.Flatten()(x)
    # Dense layer with 128 units and ReLU activation
    x = layers.Dense(MAGIC_NUMBER * 8, activation='relu')(x)
    # Dropout for regularization
    x = layers.Dropout(0.5)(x)
    # Output layer with softmax activation for classification
    output_layer = layers.Dense(classes_count, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)