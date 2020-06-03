import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from tqdm import tqdm
from copy import copy

class Compose(object):
    """Composes several transforms together.
    # Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    # Example:
        >>> Compose([
        >>>     transforms.StandartScale(),
        >>>     transforms.ToPolar(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        """
        # Returns: formatted strings with class_names, parameters and some statistics for each class
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '{0} \n'.format(t)
        format_string += '\n)'
        return format_string

class StandartScale(object):
    """Standartizes coordinates by removing the mean and scaling to unit variance
    # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        with_mean (boolean, True by default): If True, center the data before scaling
        with_std (boolean, True by default): If True, scale the data to unit variance (or equivalently, unit standard deviation).
        x_col (str, 'x' by default): X column in data
        y_col (str, 'y' by default): Y column in data
        z_col (str, 'z' by default): Z column in data
    """

    def __init__(self, drop_old=True, with_mean=True, with_std=True, x_col='x', y_col='y', z_col='z'):
        self.drop_old = drop_old
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(with_mean, with_std)
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col

    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: transformed dataframe
        """
        assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        norms = pd.DataFrame(self.scaler.fit_transform(data[[self.x_col, self.y_col, self.z_col]]))
        x_norm = norms[0]
        y_norm = norms[1]
        z_norm = norms[2]
        if self.drop_old is False:
            data = data.assign(y_old=data.y, x_old=data.x, z_old=data.z)
            data = data.assign(x=x_norm, y=y_norm, z=z_norm)
        else:
            data = data.assign(x=x_norm, y=y_norm, z=z_norm)
        return data

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with scaling_parameters: drop_old={ self.drop_old}, with_mean={self.with_mean},with_std={self.with_std} \n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f' Mean: {self.scaler.mean_} \n Var: {self.scaler.var_} \n Scale: {self.scaler.scale_} '

class MinMaxScale(object):
    """Transforms features by scaling each feature to a given range.
     # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        feature_range (Tuple (min,max), default (0,1)): Desired range of transformed data.
        x_col (str, 'x' by default): X column in data
        y_col (str, 'y' by default): Y column in data
        z_col (str, 'z' by default): Z column in data
    """

    def __init__(self, drop_old=True, feature_range=(0, 1), x_col='x', y_col='y', z_col='z'):
        self.drop_old = drop_old
        assert feature_range[0] < feature_range[1], 'minimum is not smaller value then maximum'
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col

    def __call__(self, data):
        """
        Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: transformed dataframe
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        try:
            norms = pd.DataFrame(self.scaler.fit_transform(data[[self.x_col, self.y_col, self.z_col]]))
        except IndexError:
            print(f'Columns {[self.x_col, self.y_col, self.z_col]} are not in the dataframe')
        x_norm = norms[0]
        y_norm = norms[1]
        z_norm = norms[2]
        if self.drop_old is not True:
             data = data.assign(y_old=data.y, x_old=data.x, z_old=data.z)
             data = data.assign(x=x_norm, y=y_norm, z=z_norm)
        else:
             data = data.assign(x=x_norm, y=y_norm, z=z_norm)
        return data

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, feature_range={self.feature_range} \n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f' Data min: {self.scaler.data_min_} \n Data max: {self.scaler.data_max__} \n Scale: {self.scaler.scale_} '

class Normalize(object):
    """Normalizes samples individually to unit norm.
    Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of
    other samples so that its norm (l1, l2 or inf) equals one.

      # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        norm (‘l1’, ‘l2’, or ‘max’ (‘l2’ by default)): The norm to use to normalize each non zero sample. If norm=’max’ is used, values will be rescaled by the maximum of the absolute values.
        x_col (str, 'x' by default): X column in data
        y_col (str, 'y' by default): Y column in data
        z_col (str, 'z' by default): Z column in data
    """

    def __init__(self, drop_old=True, norm='l2', x_col='x', y_col='y', z_col='z'):
        self.drop_old = drop_old
        self.norm=norm
        self.scaler = Normalizer(norm=norm)
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col

    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: transformed dataframe
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        norms = pd.DataFrame(self.scaler.fit_transform(data[[self.x_col, self.y_col, self.z_col]]))
        x_norm = norms[0]
        y_norm = norms[1]
        z_norm = norms[2]
        if self.drop_old is not True:
             data = data.assign(y_old=data.y, x_old=data.x, z_old=data.z)
             data = data.assign(x=x_norm, y=y_norm, z=z_norm)
        else:
             data = data.assign(x=x_norm, y=y_norm, z=z_norm)
        return data

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, norm={self.norm} \n' + \
               '------------------------------------------------------------------------------------------------\n'

class ConstraintsNormalize(object):
    """Normalizes samples using station characteristics.
    Each station can have its own constraints or global constrains.
      Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        columns (list or tuple of length 3): Columns to scale
        margin (number, positive): margin applied to stations (min = min-margin, max=max+margin)
        constraints (dict, None by deault) If None, constraints are computed using dataset statistics.
        use_global_constraints (boolean, True by default) If True, all data is scaled using given global constraints.

    If use_global_constraints is True and constraints is not None, constraints must be {column:(min,max)},
    else it must be {station: {column:(min,max)}}.

    Station keys must be in dataset.
    """

    def __init__(self, drop_old=True, columns=['x','y','z'], margin=1e-3, use_global_constraints=True, constraints=None):
        self.drop_old = drop_old
        assert len(columns)==3, 'Number of columns must be 3'
        self.columns = columns
        assert margin >0, 'Margin is not positive'
        self.margin = margin
        self.use_global_constraints = use_global_constraints
        self.constraints = constraints
        if constraints is not None:
            if use_global_constraints:
                for col in columns:
                    assert col in constraints.keys(), f'{col} is not in constraint keys'
                    assert len(constraints[col]) == 2, f'Not applicable number of constraints for column {col}'
                    assert constraints[col][0] < constraints[col][1], f'Minimum is not smaller than maximum for column {col}'
            else:
                for key, constraint in constraints.items():
                    for col in columns:
                        assert col in constraint.keys(), f'{col} is not in constraint keys for station {key}'
                        assert len(constraint[col]) == 2, f'Not applicable number of constraints for column {col} and station {key}'
                        assert constraint[col][0] < constraints[col][1], f'Minimum is not smaller than maximum for column {col} and station {key}'




    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: transformed dataframe
        """
        if self.constraints is None:
            self.constraints = self.get_stations_constraints(data)
        if self.use_global_constraints:
            global_constrains = {}
            for col in self.columns:
                global_min = min([x[col][0] for x in self.constraints.values()])
                global_max = max([x[col][1] for x in self.constraints.values()])
                global_constrains[col] = (global_min, global_max)
            x_norm, y_norm, z_norm = self.normalize(data, global_constrains)
            if self.drop_old is not True:
                for col in self.columns:
                    data.loc[:, col + '_old'] = data.loc[:, col]
            else:
                pass
            data.loc[:, self.columns[0]] = x_norm
            data.loc[:, self.columns[1]] = y_norm
            data.loc[:, self.columns[2]] = z_norm
        else:
            assert all([station in data['station'].unique() for station in
                        self.constraints.keys()]) is True, 'Some Station keys in constraints are not presented in data'
            if self.drop_old is not True:
                    for col in self.columns:
                        data.loc[:,col+'_old'] = data.loc[:,col]
            for station in self.constraints.keys():
                group = data.loc[data['station'] == station,]
                x_norm, y_norm, z_norm = self.normalize(group, self.constraints[station])
                data.loc[data['station'] == station, self.columns[0]] = x_norm
                data.loc[data['station'] == station, self.columns[1]] = y_norm
                data.loc[data['station'] == station, self.columns[2]] = z_norm
        return data

    def get_stations_constraints(self, df):
        groupes = df['station'].unique()
        station_constraints = {}
        for station_num in groupes:
            group = df.loc[df['station'] == station_num,]
            min_x, max_x = min(group[self.columns[0]]) - self.margin, max(group[self.columns[0]]) + self.margin
            min_y, max_y = min(group[self.columns[1]]) - self.margin, max(group[self.columns[1]]) + self.margin
            min_z, max_z = min(group[self.columns[2]]) - self.margin, max(group[self.columns[2]]) + self.margin
            station_constraints[station_num] = {self.columns[0]: (min_x, max_x),
                                                self.columns[1]: (min_y, max_y),
                                                self.columns[2]: (min_z, max_z)}
        return station_constraints

    def normalize(self, df, constraints):
        x_min, x_max = constraints[self.columns[0]]
        y_min, y_max = constraints[self.columns[1]]
        z_min, z_max = constraints[self.columns[2]]
        x_norm = 2 * (df[self.columns[0]] - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (df[self.columns[1]] - y_min) / (y_max - y_min) - 1
        z_norm = (df[self.columns[2]] - z_min) / (z_max - z_min)
        return x_norm, y_norm, z_norm

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, use_global_constraints={self.use_global_constraints} \n' + \
             f'                                             margin={self.margin}, columns={self.columns}\n '+ \
               '------------------------------------------------------------------------------------------------\n'+\
             f'constraints are: {self.constraints}'

class DropShort(object):
    """Drops tracks with num of points less then given from data.
      # Args:
        num_stations (int, default None): Desired number of stations (points). If None, maximum stations number for one track is taken from data.
        keep_misses (bool, default True): If True, points with no tracks are preserved, else they are deleted from data.
        station_column (str, 'station' by default): Event column in data
        track_column (str, 'track' by default): Track column in data
        event_column (str, 'event' by default): Station column in data
    """

    def __init__(self, num_stations=None, keep_misses=True, station_column='station', track_column='track', event_column='event'):
        self.num_stations = num_stations
        self.keep_misses = bool(keep_misses)
        self.broken_tracks_ = None
        self.num_broken_tracks_ = None
        self.station_column = station_column
        self.track_column = track_column
        self.event_column = event_column

    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: cleaned dataframe
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        if self.keep_misses is True:
            misses = data.loc[data[self.track_column] == -1, :]

        data = data.loc[data[self.track_column] != -1, :]
        tracks = data.groupby([self.event_column, self.track_column])
        if self.num_stations is None:
            self.num_stations = tracks.size().max()
        good_tracks = tracks.filter(lambda x: x.shape[0] >= self.num_stations)
        broken = list(data.loc[~data.index.isin(good_tracks.index)].index)
        self.broken_tracks_ = data.loc[broken, [self.event_column, self.track_column, self.station_column]]
        self.num_broken_tracks_ = len(self.broken_tracks_[[self.event_column, self.track_column]].drop_duplicates())

        if self.keep_misses is True:
            good_tracks = pd.concat([good_tracks, misses], axis=0).reset_index()
        return good_tracks

    def get_broken(self):
        return self.broken_tracks_

    def get_num_broken(self):
        return self.num_broken_tracks_

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: num_stations={self.num_stations}, keep_misses={self.keep_misses}, \n' + \
               f'    track_column={self.track_column}, station_column={self.station_column}, event_column={self.event_column}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Number of broken tracks: {self.num_broken_tracks_} \n'

class DropWarps(object):
    """Drops tracks with points on same stations (e.g. (2,2,2) or (1,2,1)).
      # Args:
        keep_misses (bool, default True): If True, points with no tracks are preserved, else they are deleted from data.
        station_column (str, 'station' by default): Event column in data
        track_column (str, 'track' by default): Track column in data
        event_column (str, 'event' by default): Station column in data
    """

    def __init__(self, keep_misses=True, station_column='station', track_column='track', event_column='event'):
        self.keep_misses = bool(keep_misses)
        self.broken_tracks_ = None
        self.num_broken_tracks_ = None
        self.station_column = station_column
        self.track_column = track_column
        self.event_column = event_column

    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: cleaned dataframe
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        if self.keep_misses is True:
            misses = data.loc[data[self.track_column] == -1, :]
        data = data.loc[data[self.track_column] != -1, :]
        tracks = data.groupby([self.event_column, self.track_column])
        good_tracks = tracks.filter(lambda x: x[self.station_column].unique().shape[0] == x[self.station_column].shape[0])
        broken = list(data.loc[~data.index.isin(good_tracks.index)].index)
        self.broken_tracks_ = data.loc[broken, [self.event_column, self.track_column, self.station_column]]
        self.num_broken_tracks_ = len(self.broken_tracks_[[self.event_column, self.track_column]].drop_duplicates())
        if self.keep_misses is True:
            good_tracks = pd.concat([good_tracks, misses], axis=0).reset_index()
        return good_tracks

    def get_broken(self):
        return self.broken_tracks_

    def get_num_broken(self):
        return self.num_broken_tracks_

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: keep_misses={self.keep_misses},\n    track_column={self.track_column},' + \
               f'station_column={self.station_column}, event_column={self.event_clumn}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Number of warps: {self.num_broken_tracks_} \n'

class DropMisses(object):
    """Drops points without tracks.
    Args:
        track_col (str, 'track' by default): Track column in data
    """
    def __init__(self, track_col='track'):
        self.num_misses_ = None
        self.track_col = track_col

    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: cleaned dataframe
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"

        misses = data.loc[data[self.track_col] == -1, :]
        self.num_misses_ = len(misses)
        data = data.loc[data[self.track_col] != -1, :].reset_index()
        return data

    def get_num_misses(self):
        return self.num_misses_

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: track_col={self.track_col}' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'Number of misses: {self.num_misses_} \n'

class ToPolar(object):
    """Convertes data to polar coordinates. Note that cartesian coordinates are used in reversed order!
       Formula used: r = sqrt(x^2 + y^2), phi = atan2(x,y)

       # Args:
           drop_old (boolean, False by default): If True, old coordinate features are deleted from data
           x_col (str, 'x' by default): X column in data
           y_col (str, 'y' by default): Y column in data
           z_col (str, 'z' by default): Z column in data

    """

    def __init__(self, drop_old=False, x_col='x', y_col='y', z_col='z'):
        self.drop_old = drop_old
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col

    def __call__(self, data):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data: transformed dataframe
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        r = np.sqrt(data[self.x_col] ** 2 + data[self.y_col] ** 2)
        phi = np.arctan2(data[self.x_col], data[self.y_col])
        data = data.assign(r=r, phi=phi, z=data[self.z_col])

        self.phi_range_ = (min(data.phi), max(data.phi))
        self.r_range_ = (min(data.r), max(data.r))
        self.z_range_ = (min(data.z), max(data.z))
        data = data.assign(y=data.y, x=data.x)
        if self.drop_old is True:
            del data[self.x_col]
            del data[self.y_col]
        return data

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
                f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, keep_names={self.keep_names}, ' +\
                'x_col={self.x_col}, y_col={self.y_col}, z_col={self.z_col}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f' Phi range: {self.phi_range_} \n R range: {self.r_range_} \n Z range: {self.z_range_} '


class ToCartesian(object):
    """Converts coordinates to cartesian. Formula is: y = r * cos(phi), x = r * sin(phi).
    Note that always resulting columns are x,y,z.
      # Args:
        drop_old (boolean, True by default): If True, unscaled features are dropped from dataframe
        phi_col (str, 'phi' by default): Phi column in data
        r_col (str, 'r' by default): R column in data
    """

    def __init__(self, drop_old=True, phi_col='phi', r_col='r'):
        self.drop_old = drop_old
        self.phi_col = phi_col
        self.r_col = r_col


    def __call__(self, data):
        """
        # Args:
            img (PIL Image): Image to be scaled.
        # Returns:
            PIL Image: Rescaled image.
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        y_new = data[self.r_col] * np.cos(data[self.phi_col])
        x_new = data[self.r_col] * np.sin(data[self.phi_col])
        data = data.assign(x=x_new, y=y_new)

        self.x_range_ = (min(data.x), max(data.x))
        self.y_range_ = (min(data.y), max(data.y))
        self.z_range_ = (min(data.z), max(data.z))

        if self.drop_old is True:
            del data[self.phi_col]
            del data[self.r_col]
        return data

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: drop_old={self.drop_old}, phi_col={self.phi_col}, r_col={self.r_col}\n' + \
               '------------------------------------------------------------------------------------------------\n' + \
               f'X range: {self.x_range_} \nY range: {self.y_range_} \nZ range: {self.z_range_} '

class ToBuckets(object):
    """Data may contains from tracks with varying lengths.
    To prepare a train dataset in a proper way, we have to
    split data on so-called buckets. Each bucket includes
    tracks based on their length, as we can't predict the
    6'th point of the track with length 4, but we can predict
    3-d point

        # Args:
            X: ndarray with dtype=float32 with shape of size 3
            random_state: int, seed for the RandomState
            shuffle: boolean, whether or not shuffle output dataset
            keep_misses: boolean, True by default. If True,
                     points without tracks are preserved.
    """

    def __init__(self, flat=True, shuffle=False, random_state=42, keep_misses=False):
        self.flat = flat
        self.shuffle = shuffle
        self.random_state = random_state
        self.keep_misses = keep_misses


    def __call__(self, df):
        """
        # Args:
            data: pd.DataFrame to clean up.
        # Returns:
            data (pd.DataFrame or dict(len:pd.DataFrame): transformed dataframe,
            if flat is True, returns dataframe with specific column, else dict with bucket dataframes
        """
        #assert type(data) == pd.core.frame.DataFrame, "unsupported data format"
        misses = df.loc[df['track'] == -1, ]
        df = df.loc[df['track'] != -1, ]

        rs = np.random.RandomState(self.random_state)
        groupby = df.groupby(['event', 'track'])
        maxlen = groupby.size().max()
        n_stations_min = groupby.size().min()
        subbuckets = {}
        res = {}
        val_cnt = groupby.size().unique() #get all unique track lens (in BES3 all are 3)
        for length in val_cnt:
            this_len = groupby.filter(lambda x: x.shape[0] == length)
            bucket_index = list(df.loc[df.index.isin(this_len.index)].index)
            subbuckets[length] = bucket_index
        # approximate size of the each bucket
        bsize = len(df) // (maxlen - 2)
        # set index
        k = maxlen
        buckets = {i: [] for i in subbuckets.keys()}
        # reverse loop until two points
        for n_points in range(maxlen, 2, -1):
            # while bucket is not full
            while len(buckets[n_points]) < bsize:
                if self.shuffle:
                    rs.shuffle(subbuckets[k])
                # if we can't extract all data from subbucket
                # without bsize overflow
                if len(buckets[n_points]) + len(subbuckets[k]) > bsize:
                    n_extract = bsize - len(buckets[n_points])
                    # extract n_extract samples
                    buckets[n_points].extend(subbuckets[k][:n_extract])
                    # remove them from original subbucket
                    subbuckets[k] = subbuckets[k][n_extract:]
                else:
                    buckets[n_points].extend(subbuckets[k])
                    # remove all data from the original list
                    subbuckets[k] = []
                    # decrement index
                    k -= 1
        self.buckets_ = buckets
        if self.flat is True:
            res = copy(df)
            res['bucket'] = 0
            for i, bucket in buckets.items():
                res.loc[bucket, 'bucket'] = i
            if self.keep_misses:
                misses.loc[:,'bucket'] = -1
                res = pd.concat([res, misses], axis=0)
        else:
            res = {i: df.loc[bucket] for i, bucket in buckets.items()}
            if self.keep_misses:
                res[-1] = misses
        return res

    def get_bucket_index(self):
        """
        # Returns: dict(len: indexes) - dict with lens and list of indexes in bucket
        """
        return self.buckets_

    def get_buckets_sizes(self):
        """

        # Returns:
            {bucket:len} dict with length of data in bucket
        """
        return {i: len(j) for i,j in self.buckets_.items()}

    def __repr__(self):
        return '------------------------------------------------------------------------------------------------\n' + \
               f'{self.__class__.__name__} with parameters: flat={self.flat}, random_state={self.random_state}, shuffle={self.shuffle}, keep_misses={self.keep_misses}\n' + \
               '------------------------------------------------------------------------------------------------\n'



