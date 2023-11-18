import tensorflow as tf

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model


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
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(time_steps, activation='softmax')(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = layers.Multiply()([inputs, a_probs])
    return output_attention_mul

def create_3dcnn_lstm_attention_djamaco_model(input_shape, classes_count):
    input_layer = layers.Input(input_shape)

    MAGIC_NUMBER = 16

    x = layers.Conv3D(MAGIC_NUMBER, (3, 3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(MAGIC_NUMBER * 2, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool3D(pool_size=(1, 2, 2))(x)
    x = layers.BatchNormalization()(x)

    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.LSTM(MAGIC_NUMBER * 4, return_sequences=True)(x)

    x = get_attention_based(x, input_shape[0])

    x = layers.Flatten()(x)
    x = layers.Dense(MAGIC_NUMBER * 8, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(classes_count, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output_layer)