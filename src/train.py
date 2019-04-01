import tensorflow as tf
import numpy as np
import plac 

from tensorflow.keras.optimizers import Adam

# tracknet utils
from tracknet import build_tracknet_model
from losses import tracknet_loss
from metrics import circle_area
from metrics import point_in_ellipse
from metrics import MetricsCallback


@plac.annotations(
    data_path=("Path to the .npz file with training and validation data", "option", None, str),
    batch_size=("The size of the batch", "option", None, int),
    random_seed=("Seed for the random generator", "option", None, int))
def main(data_path='data/train_dataset.npz', batch_size=512, random_seed=13):
    # set random seed for reproducible results
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    print("Read data")
    data = np.load(data_path)
    print("Train size: %d" % len(data['x_train']))
    print("Validation size: %d" % len(data['x_val']))

    print("\nCreate and compile TrackNet model")
    tracknet = build_tracknet_model(data['x_train'].shape[-1])
    # print network summary
    tracknet.summary()
    # compile keras model
    tracknet.compile(
        loss=tracknet_loss(),
        optimizer=Adam(clipnorm=1.),
        metrics=[point_in_ellipse, circle_area])

    # train the network
    metrics_cb = MetricsCallback(test_data=data['full_val'])
    print("Training...")
    history = tracknet.fit(x=data['x_train'],
                           y=data['y_train'],
                           batch_size=batch_size,
                           epochs=50,
                           callbacks=[metrics_cb],
                           validation_data=(
                               data['x_val'], 
                               data['y_val']))
    #print("Save model's weights")
    #tracknet.save_weights('data/tracknet_weights.h5')



if __name__ == "__main__":
    plac.call(main)