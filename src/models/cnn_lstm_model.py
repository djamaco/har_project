import tensorflow as tf
from src.config import FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH

layers = tf.keras.layers
Sequential = tf.keras.models.Sequential

def create_cnn_lstm_sylvia_model(classes_count):
    """
    Source: https://www.kaggle.com/code/sylvianclee/human-activity-recognition-cnn-lstm
    Creates a CNN-LSTM model. 

    Args:
        classes_count (int): The number of classes in the dataset.

    Returns:
        keras.models.Sequential: The CNN-LSTM model.
    """
    model = Sequential()
    model.add(layers.Reshape((FRAMES_COUNT, IMAGE_HEIGHT * IMAGE_WIDTH , 3), input_shape=(FRAMES_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
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