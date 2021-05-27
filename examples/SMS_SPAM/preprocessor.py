import sys
sys.path.append('../../')

from spear.labeling import preprocessor

@preprocessor()
def convert_to_lower(x):
    return x.lower().strip()