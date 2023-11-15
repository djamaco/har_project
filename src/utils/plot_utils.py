import os
import matplotlib.pyplot as plt
from src.config import MODELS_DIR

def create_plot_metric_and_save_to_model(model_name, model_training_history, metric_name_1, metric_name_2, plot_name):
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    epochs = range(len(metric_value_1))
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    
    plt.savefig(os.path.join(MODELS_DIR, model_name, f'{plot_name}.png'))
    plt.clf()
    plt.cla()
    plt.close()