from typing import Union

from src.config.constants import ModelName
from .convlstm_model import create_convlstm_bleed_model
from .lrcn_model import create_lrcn_djamaco_model, create_lrcn_bleed_model
from .cnn_lstm_model import create_cnn_lstm_sylvia_model, create_3dcnn_lstm_attention_djamaco_model
from .r21d_model import create_r21d_djamaco_model

import tensorflow as tf

Sequential = tf.keras.models.Sequential

models_create_functions = {
    ModelName.CONVLSTM_BLEED: create_convlstm_bleed_model,
    ModelName.LRCN_DJAMACO: create_lrcn_djamaco_model,
    ModelName.LRCN_BLEED: create_lrcn_bleed_model,
    ModelName.CNN_LSTM_SYLVIA: create_cnn_lstm_sylvia_model,
    ModelName.R21D_DJAMACO: create_r21d_djamaco_model,
    ModelName.CNN_LSTM_ATTENTION_DJAMACO: create_3dcnn_lstm_attention_djamaco_model,
}

def get_model(model_name: Union[ModelName, str], **kwargs) -> Sequential:
    # Convert string to ModelName enum if necessary
    if isinstance(model_name, str):
        try:
            model_name = ModelName(model_name)
        except ValueError:
            raise ValueError(f'Invalid model name string: {model_name}')
    # Get the model creation function
    model_create_function = models_create_functions.get(model_name)
    if model_create_function is None:
        raise ValueError(f'No model found: {model_name}')
    model = model_create_function(**kwargs)
    # Display the models summary.
    model.summary()
    return model
    

