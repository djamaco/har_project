import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='A name of a choosen deep learning model that we wanna run.', type=str)
parser.add_argument('--zip', '-z', help='A flag that indicates if we want to zip the model or not.', type=str)

def get_args():
    args, unknown = parser.parse_known_args()
    # A possible args preprocessing can be added here
    return args, unknown
