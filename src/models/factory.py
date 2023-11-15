from typing import Union

from src.config.constants import ModelName
from .convlstm_model import create_convlstm_model
from .lrcn_model import create_lrcn_model

import tensorflow as tf

Sequential = tf.keras.models.Sequential

models_create_functions = {
    ModelName.CONVLSTM: create_convlstm_model,
    ModelName.LRCN: create_lrcn_model,
}

def get_model(model_name: Union[ModelName, str], **kwargs) -> Sequential:
    model_create_function = models_create_functions.get(model_name if isinstance(model_name, ModelName) else ModelName(model_name))
    if model_create_function is None:
        raise ValueError(f'No model found: {model_name}')
    model = model_create_function(**kwargs)
    # Display the models summary.
    model.summary()
    return model
    

