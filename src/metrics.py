from keras.callbacks import Callback
from .data_utils import get_part
import tensorflow as tf
import numpy as np
import math
import time


def circle_area(y_true, y_pred):
    def area(R1, R2):
        return R1*R2*math.pi
    areas = area(y_pred[:, 2], y_pred[:, 3])
    return areas


def point_in_ellipse_numpy(y_true, y_pred):
    # checks if the next point of seed 
    # is included in predicted circle
    # 1 - if yes, 0 - otherwise
    # x coords
    x_coord_true = y_true[:, 0]
    x_coord_pred = y_pred[:, 0]
    # y coords
    y_coord_true = y_true[:, 1]
    y_coord_pred = y_pred[:, 1]
    # coordinate's distances
    x_dist = np.square(x_coord_pred - x_coord_true)
    y_dist = np.square(y_coord_pred - y_coord_true)
    # calculate x and y parts of equation
    x_part = x_dist / np.square(y_pred[:, 2])
    y_part = y_dist / np.square(y_pred[:, 3])
    # left size of equation x_part + y_part = 1
    left_size = x_part + y_part
    # if left size less than 1, than point in ellipse
    return np.less_equal(left_size, 1)


def point_in_ellipse(y_true, y_pred):
    # checks if the next point of seed 
    # is included in predicted circle
    # 1 - if yes, 0 - otherwise
    # x coords
    x_coord_true = y_true[:, 0]
    x_coord_pred = y_pred[:, 0]
    # y coords
    y_coord_true = y_true[:, 1]
    y_coord_pred = y_pred[:, 1]
    # coordinate's distances
    x_dist = tf.square(x_coord_pred - x_coord_true)
    y_dist = tf.square(y_coord_pred - y_coord_true)
    # calculate x and y parts of equation
    x_part = x_dist / tf.square(y_pred[:, 2])
    y_part = y_dist / tf.square(y_pred[:, 3])
    # left size of equation x_part + y_part = 1
    left_size = x_part + y_part
    # if left size less than 1, than point in ellipse
    return tf.less_equal(left_size, 1)


class MetricsCallback(Callback):
    def __init__(self, test_data):
        self.x_test = test_data

    def on_train_begin(self, logs={}):
        self.efficiency = []
        self.hits_efficiency = []

    def on_epoch_end(self, epoch, logs={}):
        start_time = time.time()
        efficiency = 0
        hits_efficiency = 0
        x_val = self.x_test.copy()
        # run from the first station to the last
        for i in range(2, x_val.shape[1]):
            # cut off all track-candidates
            x_part, target = get_part(x_val, i)
            # get model's prediction
            preds = self.model.predict(x_part, batch_size=512)
            # get indices of the tracks, which continuation was found
            idx = point_in_ellipse_numpy(target, preds)
            # count number of right predictions
            hits_efficiency += np.sum(idx) / len(idx)
            # exclude 
            x_val = x_val[idx]
        # count number of track for which we found all points
        efficiency = len(x_val) / len(self.x_test)
        # recompute the hits_efficiency
        hits_efficiency /= x_val.shape[1] - 2
        # add metrics to the list
        self.hits_efficiency.append(hits_efficiency)
        self.efficiency.append(efficiency)
        # time in seconds
        end_time = time.time() - start_time
        # processing speed
        proc_speed = len(self.x_test) / end_time
        # print metrics
        print('Efficiency: %.4f - Hits efficiency: %.4f - Processing speed: %.2f tracks/sec' % 
            (efficiency, hits_efficiency, proc_speed))
