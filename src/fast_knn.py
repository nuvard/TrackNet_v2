from sklearn import preprocessing
import numpy as np


def knn(x_train, x_test, n_neighbours=1, return_similarities=False):
    '''Implements Fast K-Nearest Neighbours algorithm
    Based on the method used in "Learning To Remember Rare Events"
    by Lukasz Kaiser, Ofir Nachun, Aurko Roy, and Samy Bengio
    Paper: https://openreview.net/pdf?id=SJTQLdqlg

    # Arguments
        x_train: ndarray, dtype=float32, train array
        x_test: ndarray, dtype=float32, test points
        n_neighbours: int, optional. Number of nearest neighbours to find
        return_similarities: boolean, optional. Whether or not return top_K similarities
    '''
    x_train_l2 = preprocessing.normalize(x_train, norm='l2')
    x_test_l2 = preprocessing.normalize(x_test, norm='l2')
    similarity_matrix = np.matmul(x_test_l2, x_train_l2.T)
    # similarities, indices of the max sims
    sims, closest_idx = top_k(similarity_matrix, k=n_neighbours)

    if return_similarities:
        return closest_idx, sims

    return closest_idx


def top_k(inputs, k=1):
    '''Finds top elements for along the last dimension
    of the inputs (along each row for matrices)

    # Arguments
        input: 1-D or 2-D array with last dimension at least k
        k: int, number of elements to extract

    # Returns
        values, indices
    '''
    assert len(inputs.shape) < 3, "Input must have at least 2 dimensions"
    assert k >= 1, "Got k=%d, the value greater than 0 is required" % k
    assert type(k) == int, "The type of k is %s, int is required" % str(type(k))

    if len(inputs.shape) == 1:
        # if vector -> to matrix
        inputs = np.expand_dims(inputs, 0)
    # need for indexing
    i = np.arange(len(inputs))[:, np.newaxis]
    # sort in descending order
    indices = np.argsort(inputs, axis=-1)[:, ::-1]
    # get top-k indices
    indices = indices[:, :k]
    # get top-k values
    values = inputs[i, indices]
    return values, indices