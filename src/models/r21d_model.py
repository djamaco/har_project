import tensorflow as tf

Sequential = tf.keras.models.Sequential
layers = tf.keras.layers

def create_r21d_djamaco_model(input_shape, classes_count):
    """
    NOT IMPLEMENTED.
    """
    model = Sequential()

    filters_number = 64
    kernel_number = 3

    model.add(layers.TimeDistributed(layers.Conv2D(filters=filters_number, kernel_size=(kernel_number, kernel_number), padding='same'), input_shape=input_shape))
    model.add(layers.Reshape((input_shape[0], input_shape[1] * input_shape[2], filters_number)))
    model.add(layers.Conv1D(filters=filters_number, kernel_size=kernel_number, padding='same'))
    model.add(layers.Reshape((input_shape + (64,))))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2)))


    # model.add(layers.TimeDistributed(layers.Conv2D(filters=filters_number, kernel_size=(3, 3), activation='relu'), input_shape=input_shape))
    # # model.add(layers.TimeDistributed(layers.BatchNormalization()))
    # model.add(layers.Reshape((input_shape[0], input_shape[1] * input_shape[2], filters_number)))
    # model.add(layers.Conv1D(filters=filters_number, kernel_size=kernel_number, activation='relu'))
    # # model.add(layers.BatchNormalization())

    # model.add(layers.TimeDistributed(layers.Conv2D(filters=filters_number, kernel_size=(kernel_number, kernel_number), activation='relu'), input_shape=input_shape))
    # # model.add(layers.TimeDistributed(layers.BatchNormalization()))
    # model.add(layers.Reshape((input_shape[0], input_shape[1] * input_shape[2], filters_number)))
    # model.add(layers.Conv1D(filters=filters_number, kernel_size=kernel_number, activation='relu'))
    # # model.add(layers.BatchNormalization())

   
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(classes_count, activation='softmax'))

    return model