import pandas as pd
import numpy as np
import json
import os

from glob import glob
from tqdm import tqdm
tqdm.pandas()

from timing import timeit


def read_vertex_file(path):
    with open(path) as f:
        return json.load(f)


def sample_vertices(vertex_stats, n, random_seed=None):
    rs = np.random.RandomState(seed=random_seed)
    # sample vertex X coordinate
    vertex_x = rs.normal(
        loc=vertex_stats['x']['mean'], 
        scale=vertex_stats['x']['std'],
        size=(n, 1))
    # sample vertex Y coordinate
    vertex_y = rs.normal(
        loc=vertex_stats['y']['mean'], 
        scale=vertex_stats['y']['std'],
        size=(n, 1))
    # sample vertex Z coordinate
    vertex_z = rs.uniform(
        low=vertex_stats['z']['left'],
        high=vertex_stats['z']['right'],
        size=(n, 1))
    # return stacked coordinates
    return np.hstack([vertex_x, vertex_y, vertex_z])


def get_tracks_with_vertex(df, vertex_stats, random_seed=13):
    # extract tracks groups
    groupby = df.groupby(['event', 'track'])
    # sample vertex data
    vertex = sample_vertices(vertex_stats, groupby.ngroups, random_seed)
    # create result array
    n_stations = groupby.size().max()
    # vertex + n_stations 3D hits
    res = np.zeros((groupby.ngroups, n_stations+1, 3))
    # fill with vertex data
    res[:, 0] = vertex
    # get tracks
    tracks = groupby[['x', 'y', 'z']].progress_apply(pd.Series.tolist)
    # fill result array
    for i, track in enumerate(tqdm(tracks)):
        res[i, 1:len(track)+1] = np.asarray(track)
    
    return res
    

@timeit
def read_train_dataset(dirpath, seed_for_vertex_gen=13):
    '''Reads data from directory. Directory must have 
    the following structure:

        dir
        |__file1.tsv
        |__file2.tsv
        |...
        |__fileN.tsv

    where .tsv files contains events hits
    or with corresponding track labels
    '''
    # collect files
    train_files = glob(os.path.join(dirpath, '*.tsv'))
    vertex_file = os.path.join(dirpath, "vertex.json")

    # get vertex statistics
    vertex_stats = read_vertex_file(vertex_file)

    # get train data
    length = [None]*len(train_files)
    train = [None]*len(train_files)
    for i, f in enumerate(train_files):
        print("Processing `%s` file..." % f)
        df = pd.read_csv(f, encoding='utf-8', sep='\t')
        # extract only true tracks
        df = df[df.track != -1]
        # get true tracks array (N, 7, 3)
        train[i] = get_tracks_with_vertex(df, vertex_stats)
        length[i] = len(train[i])

    # create result array
    N = sum(length)
    # empty is faster than zeros
    train_arr = np.empty((N, *train[0].shape[1:]), dtype=np.float32)

    # record all to train_arr
    while train:
        train_arr[:length.pop()] = train.pop()

    return train_arr


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