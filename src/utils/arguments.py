import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='A name of a choosen deep learning model that we wanna run.', type=str)

def get_args():
    args, unknown = parser.parse_known_args()
    # A possible args preprocessing can be added here
    return args, unknown
