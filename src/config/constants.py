from enum import Enum

class ModelName(Enum):
    CONVLSTM = 'convlstm'
    CONVLSTM_BLEED = 'convlstm_bleed'
    LRCN_DJAMACO = 'lrcn_djamaco'
    LRCN_BLEED = 'lrcn_bleed'
    CNN_LSTM_SYLVIA = 'cnn_lstm_sylvia'
    R21D_DJAMACO = 'r21d_djamaco'
    CNN_LSTM_ATTENTION_DJAMACO = 'cnn_lstm_attention_djamaco'
