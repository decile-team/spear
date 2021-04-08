from labeling.preprocess import *

@preprocessor()
def convert_to_lower(x):
    return x.lower().strip()