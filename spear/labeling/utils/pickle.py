import pickle

def dump_to_pickle(filename: str, data):
    """Utility for dumping data into a pickle file

    Args:
        filename (str): Name of pickle file.
        data ([type]): List of objects to dump into pickle file.
    """
    f=open(filename, "wb")
    for item in data:
        pickle.dump(item,f)
    f.close()