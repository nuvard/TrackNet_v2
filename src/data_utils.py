import numpy as np
import os

def read_data(dirpath):
    '''Reads data from directory. Directory must have 
    the following structure:

        dir
        |__ X_train.npy
        |__ X_test.npy
        |__ y_train.npy
        |__ y_test.npy

    where .npy files contains track-candidates (X-files)
    or corresponding labels (y-files)
    '''
    x_train = np.load(os.path.join(dirpath, "X_train.npy"))
    x_test = np.load(os.path.join(dirpath, "X_test.npy"))
    y_train = np.load(os.path.join(dirpath, "y_train.npy"))
    y_test = np.load(os.path.join(dirpath, "y_test.npy"))
    return (x_train, y_train), (x_test, y_test)


def shuffle_arrays(*args, seed=None):
    '''Shuffles input lists of arrays if they are equal length

    # Arguments
        args : lists or numpy arrays
        seed : int, seed for the RandomState

    # Returns 
        shuffled arrays
    '''
    assert len(set(len(x) for x in args)) == 1, 'Arrays must be the same size'
    # create seed for random generator
    rs = np.random.RandomState(seed=seed)
    idx = rs.permutation(len(args[0]))
    args = [x[idx] for x in args]
    return args