import tensorflow as tf


def ellipse_loss(y_true, y_pred):
    '''Penalizes predictions when a true point 
    is far from the center of the predicted ellipse
    '''
    # x part
    x = (y_pred[:, 0] - y_true[:, 0]) / y_pred[:, 2]
    x = tf.square(x)
    # y part
    y = (y_pred[:, 1] - y_true[:, 1]) / y_pred[:, 3]
    y = tf.square(y)
    return tf.sqrt(x + y)


def radius_penalize(y_pred):
    '''Penalizes for ellipses with large radii
    '''
    return y_pred[:, 2] * y_pred[:, 3]


def tracknet_loss(lambda1 = 0.9, lambda2 = 0.1):
    # latent function
    def _tracknet_loss(y_true, y_pred):
        ellipse_part = lambda1 * ellipse_loss(y_true, y_pred)
        radius_part = lambda2 * radius_penalize(y_pred) 
        # summarize
        loss_value = ellipse_part + radius_part
        loss_value = tf.reduce_mean(loss_value, axis=0)
        return loss_value
    return _tracknet_loss


def custom_loss(y_true, y_pred):
    # TODO: lambdas as arguments
    ellipse_part = 0.9 * ellipse_loss(y_true, y_pred)
    radius_part = 0.1 * radius_penalize(y_pred) 
    # summarize
    loss_value = ellipse_part + radius_part
    loss_value = tf.reduce_mean(loss_value, axis=0)
    return loss_value