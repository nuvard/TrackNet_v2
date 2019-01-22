# data utils
from data_utils import batch_generator
from data_utils import get_dataset
from data_utils import read_data
# tracknet utils
from tracknet import build_tracknet_model
from losses import tracknet_loss
from metrics import circle_area
from metrics import point_in_ellipse
from metrics import MetricsCallback
# other
from keras.optimizers import Adam
import logging
import plac # for arguments parsing
# setup logging
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)-20s (line: %(lineno)-3s) [%(levelname)s]: %(message)s")


@plac.annotations(
data_dir=("Path to the directory containing files with data for training and testing", "positional", None, str),
batch_size=("The size of the batch", "option", None, int),
n_gpus=("Number of GPU cores for training and testing", "option", None, int),
random_seed=("Seed for the random generator", "option", None, int))
def main(data_dir, batch_size=32, n_gpus=0, random_seed=None):
    logging.info("Read data")
    x_train, x_test = read_data(data_dir)
    logging.info("Train size: %d" % len(x_train))
    logging.info("Test size: %d" % len(x_test))
    logging.info("Create batch generator")
    train_gen = batch_generator(
        X=x_train, 
        batch_size=batch_size, 
        shuffle=True, 
        random_seed=random_seed)
    # validation set
    validation = get_dataset(x_test, shuffle=True, random_seed=random_seed)
    logging.info("Create and compile TrackNet model")
    tracknet = build_tracknet_model(validation[0].shape[-1])
    logging.info(tracknet.summary())
    tracknet.compile(
        loss=tracknet_loss(),
        optimizer=Adam(clipnorm=1.),
        metrics=[point_in_ellipse, circle_area])
    # train the network
    metrics_cb = MetricsCallback(test_data=x_test)
    logging.info("Training...")
    history = tracknet.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(x_train) // batch_size,
        epochs=1,
        validation_data=validation,
        validation_steps=1,
        callbacks=[metrics_cb])
    logging.info("Save model's weights")
    tracknet.save_weights('data/tracknet_weights.h5')



if __name__ == "__main__":
    plac.call(main)