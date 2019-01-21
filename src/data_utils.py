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
    # extract only true tracks
    x_train = x_train[y_train==1]
    x_test = x_test[y_test==1]
    return (x_train, x_test)


def shuffle_arrays(*args, random_seed=None):
    '''Shuffles input lists of arrays if they are equal length

    # Arguments
        args : lists or numpy arrays
        random_seed : int, seed for the RandomState

    # Returns 
        shuffled arrays
    '''
    assert len(set(len(x) for x in args)) == 1, 'Arrays must be the same size'
    # create seed for random generator
    rs = np.random.RandomState(seed=random_seed)
    idx = rs.permutation(len(args[0]))
    args = [x[idx] for x in args]
    return args


def get_part(X, n_hits):
    '''Cuts the input array X with the shape of (N, S, 3) with 
    S points to the n_hits points and set the n_hits+1 point as target.
    Also, concatenates the X[:, i+1, 2], which is actually a z+1 coordinate,
    to the array

    # Arguments
        X : ndarray with dtype=float32 with shape of size 3
        n_hits : int, number of hits to cut off
    '''
    assert len(X.shape) == 3, "Input array must be 3-dimensional, but got {}".format(X.shape)
    assert X.shape[1] > 2, "Input shape is {}, second dimension must be greater than 2".format(X.shape)
    assert X.shape[2] == 3, "Input shape is {}, third dimension must be 3 for (x, y, z) coords".format(X.shape)
    assert n_hits < X.shape[1], "Number of hits to cut off must be smaller than the actual number of hits(%d), but got %d" % (X.shape[1], n_hits)
    x_part = np.zeros(shape=(*X.shape[:2], 4))
    # :2 - we take only (x, y) coords
    target = X[:, n_hits, :2].copy()
    # cut to n_hits
    x_part[:, :n_hits, :3] = X[:, :n_hits]
    # z+1 point
    x_part[:, :n_hits, 3] = X[:, 1:n_hits+1, 2]
    return (x_part, target)


def get_dataset(X, shuffle=False, random_seed=None):
    '''Creates a dataset to train the tracknet model. 
    The input is cropped tracks and target is the next point in track

    # Arguments:
        X : ndarray with dtype=float32 with shape of size 3
        random_seed : int, seed for the RandomState
        shuffle : boolean, whether or not shuffle output dataset
    '''
    # create pseudo random generator 
    rs = np.random.RandomState(seed=random_seed)
    # shuffle input array
    X = rs.permutation(X)
    # parts with different number of points
    n_parts = X.shape[1] - 2
    # size of each part of dataset
    # we uniformly split the input array 
    part_size = len(X) // n_parts 
    # create vars for resulting arrays
    inputs = np.zeros(shape=(n_parts*part_size, X.shape[1], 4))
    target = np.zeros(shape=(n_parts*part_size, 2))
    # create parts
    for i in range(n_parts):
        # calculate indices
        i_s = i*part_size       # start index 
        i_e = (i+1)*part_size   # end index
        # create part
        inputs[i_s:i_e], target[i_s:i_e] = get_part(X[i_s:i_e], n_hits=i+2)
    # shuffle arrays
    if shuffle:
        inputs, target = shuffle_arrays(inputs, target, random_seed=random_seed)
    
    return (inputs, target)


def batch_generator(X, batch_size, shuffle=False, random_seed=None):
    '''Batch generator function

    # Arguments:
        X : ndarray with dtype=float32 with shape of size 3
        batch_size : int, the size of the batch
        random_seed : int, seed for the RandomState
        shuffle : boolean, whether or not shuffle output dataset
    '''
    # transform input sequences into dataset
    inputs, target = get_dataset(X, shuffle=shuffle, random_seed=random_seed)
    # generate batches
    while True:
        # this loop produces batches
        for b in range(len(inputs) // batch_size):
            batch_xs = inputs[b*batch_size : (b+1)*batch_size]
            batch_ys = target[b*batch_size : (b+1)*batch_size]
            yield (batch_xs, batch_ys)