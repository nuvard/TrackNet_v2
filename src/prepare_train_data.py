import numpy as np
import plac
import os

from .data_utils import read_train_dataset
from .data_utils import train_test_split
from .data_utils import get_dataset


@plac.annotations(
    train_dir=("Path to the directory with the train data", "option", None, str),
    fname_to_save=("Name of the file to save results", "option", None, str),
    val_size=("Fraction of the validation subset of data", "option", None, float),
    random_seed=("Random seed", "option", None, int))
def main(train_dir="data/train/", 
         fname_to_save="train_dataset_vertex.npz", 
         val_size=0.2, 
         random_seed=13):
    print("Read train data")
    train_data = read_train_dataset(
        train_dir, 
        vertex_fname="vertex.json", 
        random_seed=random_seed)

    print("\nSplit on train and validation")
    train, validation = train_test_split(
        train_data, 
        test_size=val_size, 
        random_seed=random_seed)
    print("Train shape: {}".format(train.shape))
    print("Validation shape: {}".format(validation.shape))

    print("\nPrepare data for full validation")
    tracklens = np.count_nonzero(validation, axis=1)[:, -1]
    full_val_data = validation[tracklens == validation.shape[1]]

    print("\nPrepare data as input to NN")
    train = get_dataset(train, shuffle=True, random_seed=random_seed)
    validation = get_dataset(validation, shuffle=True, random_seed=random_seed)
    
    print("\nSave to the file `%s`" % fname_to_save)
    np.savez(os.path.join("data", fname_to_save),
             x_train=train[0],
             y_train=train[1],
             x_val=validation[0],
             y_val=validation[1],
             full_val=full_val_data)


if __name__ == "__main__":
    plac.call(main)