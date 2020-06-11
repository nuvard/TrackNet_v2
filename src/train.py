import tensorflow as tf
import numpy as np
import plac
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# tracknet utils
from src.tracknet import tracknet_builder
from src.losses import tracknet_loss
from src.metrics import circle_area
from src.metrics import point_in_ellipse
from src.metrics import MetricsCallback
from preprocessing import Compose, ToCylindrical, StandartScale, DropFakes
def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)


@plac.annotations(
    config_path=("Path to the config file", "option", None, str))

def main(config_path='configs/train_dropped_broken.yaml'):
    config = load_config(config_path)
    random_seed = config['random_seed']
    data_path = config['data_path']
    batch_size = config['batch_size']
    autosave = config['autosave']
    epochs = config['epochs']
    # set random seed for reproducible results
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    print("Read data")
    data = pd.read_csv(data_path)
    transform = Compose([
        DropFakes(),
        StandartScale(),
        ToCylindrical(),
        StandartScale()])
    data = transform(data)
    print("Train size: %d" % len(data['x_train']))
    print("Validation size: %d" % len(data['x_val']))

    print("\nCreate and compile TrackNet model")
    tracknet = tracknet_builder(data['x_train'].shape[-1])
    # print network summary
    tracknet.summary()
    # compile keras model
    tracknet.compile(
        loss=tracknet_loss(),
        optimizer=Adam(clipnorm=1.),
        metrics=[point_in_ellipse, circle_area])

    metrics_cb = MetricsCallback(test_data=data['full_val'])
    callbacks = [metrics_cb]

    if autosave and autosave['enabled']:
        file_prefix = autosave['file_prefix']
        directory = autosave['output_dir']
        from time import gmtime, strftime
        postfix = strftime("_%Y-%m-%d__%H.%M.%S", gmtime())
        res_name = directory + postfix
        if not os.path.exists(res_name):
            os.makedirs(res_name)
        metric_monitor = autosave['metric_monitor']
        filepath = res_name + '/' + file_prefix + "_init-{epoch:02d}-."+ metric_monitor + ".{" + metric_monitor + ":.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor=metric_monitor, verbose=1, save_best_only=True, save_weights_only=False, mode='max')
        callbacks.append(checkpoint)

    print("Training...")
    # train the network
    history = tracknet.fit(x=data['x_train'],
                           y=data['y_train'],
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=callbacks,
                           validation_data=(
                               data['x_val'], 
                               data['y_val']))
    #print("Save model's weights")
    #tracknet.save_weights('data/tracknet_weights.h5')



if __name__ == "__main__":
    plac.call(main)