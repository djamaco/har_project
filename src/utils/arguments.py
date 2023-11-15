import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='A name of a choosen deep learning model that we wanna run. For some scripts it can be a name of a trained model.', type=str)
parser.add_argument('--zip', '-z', help='A flag that indicates if we want to zip the model or not.', type=str)
parser.add_argument('--yturl', '-y', help='A youtube video url that we want to download.', type=str)

def get_args():
    args, unknown = parser.parse_known_args()
    # A possible args preprocessing can be added here
    return args, unknown
