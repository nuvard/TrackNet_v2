import tensorflow as tf
import math

def circle_area(y_true, y_pred):
    def area(R1, R2):
        return R1*R2*math.pi
    areas = area(y_pred[:, 2], y_pred[:, 3])
    return areas


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