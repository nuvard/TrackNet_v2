from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import softplus


def build_tracknet_model(n_features):
    '''Build TrackNet_v2 model based on the input_shape

    # Arguments
        input_shape : the size of the input tensor
            input tensor [N, D] 
            N = sequence length, 
            D = 4 - x, y, z, z+1 coords 
    
    # Returns 
        Keras Model 
    '''
    # `None` means that an input sequence can be an arbitrary length
    input_shape = (None, n_features)
    inputs = Input(shape=input_shape, name="inputs")
    # encode each timestep independently skipping zeros strings
    x = TimeDistributed(
        Masking(mask_value=0., name="mask"), 
        name="time_distributed_mask")(inputs)
    # timesteps encoder layer
    x = Conv1D(32, 3, padding='same', name="conv1d")(x)
    x = BatchNormalization(name='batch_norm')(x)
    x = Activation('relu', name='conv1d_relu')(x)
    # recurrent layers
    x = GRU(32, return_sequences=True, name="gru1")(x)
    x = GRU(16, name="gru2")(x)
    # outputs
    # x and y coords - centre of observing
    # area on the next station
    xy_coords = Dense(2, activation='linear', name="xy_coords")(x)
    # ellipse radii
    r1_r2 = Dense(2, activation=softplus, name="r1_r2")(x)
    outputs = Concatenate(name="outputs")([xy_coords, r1_r2])
    # create model
    tracknet = Model(inputs, outputs, name="TrackNet_v2")
    return tracknet