from typing import Union

from src.config.constants import ModelName
from .convlstm_model import create_convlstm_model_bleed
from .lrcn_model import create_lrcn_model, create_lrcn_model_bleed

import tensorflow as tf

Sequential = tf.keras.models.Sequential

models_create_functions = {
    ModelName.CONVLSTM_BLEED: create_convlstm_model_bleed,
    ModelName.LRCN: create_lrcn_model,
    ModelName.LRCN_BLEED: create_lrcn_model_bleed,
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
    

