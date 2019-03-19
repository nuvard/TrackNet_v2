import pandas as pd
import numpy as np
import warnings
import plac
import time
import os

from glob import glob
from tqdm import tqdm

from timing import timeit

warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()


def euclidean(x, y):
    '''Compute the euclidean distance 
    between two vectors or 2d arrays
    '''
    return np.sum(np.square(x-y), axis=-1)


def brute_force_1nn(x, y):
    '''K-nearest neighbors algorithm for k=1
    using brute-force
    '''
    return [np.argmin(euclidean(x, y_)) for y_ in y]


def match_hits_to_mc_points(event_id, 
                            event_hits, 
                            event_mc_points, 
                            mc_track_ids):
    '''For each hit finds the most closest Monte-Carlo
    point and assigns to it's track_id to this hit.
    
    Note: 
        one hit may contain two mc_points, 
        such hit will be duplicated

    # Arguments
        event_id: int, identifier of the event
        event_hits: ndarray, dtype=float, event's hits
        event_mc_points: Monte-Carlo points
        mc_track_ids: track identifiers

    # Returns
        array with shape=(N, 5), where columns are:
        event_id, x, y, z, track_id
    '''
    # Faster than KD-Tree and avoids dynamic allocations
    ii = brute_force_1nn(event_hits, event_mc_points)
    # indices of hits without track_id
    ii_ = set(range(len(event_hits))) - set(ii)
    ii_ = list(ii_)
    # event_id, x, y, z, track_id
    # track_id = -1, if fake hit
    result_array = np.full((len(ii)+len(ii_), 5), -1, dtype=np.float32)
    # add event_id
    result_array[:, 0] = event_id
    # get hits with track_ids
    result_array[:len(ii), 1:4] = event_hits[ii].copy()
    result_array[:len(ii), -1] = mc_track_ids.copy()
    # get the remaining hits
    result_array[len(ii):, 1:4] = event_hits[ii_].copy()
    return result_array


@timeit
def read_mc_file(mc_fpath, sep='\t', index_col=None, encoding='utf-8'):
    '''Reads file 'mc_fpath'
    
    # Return
        pandas.DataFrame{event, track, x, y, z, station}
    '''
    usecols = ['event', 'track', 'x_in', 'y_in', 'z_in', 'station']
    dtypes = [np.int32, np.int32, np.float32, np.float32, np.float32, np.int32]
    # read dataframe
    df = pd.read_csv(mc_fpath, sep=sep, encoding=encoding, index_col=index_col, 
                     usecols=usecols, dtype=dict(zip(usecols, dtypes)))
    # rename columns
    df = df.rename(columns={'x_in': 'x', 'y_in': 'y', 'z_in': 'z'})
    return df


@timeit
def read_hits_file(hits_fpath, sep='\t', index_col=0, encoding='utf-8'):
    '''Reads file 'mc_fpath'
    
    # Return
        pandas.DataFrame{event, x, y, z, station}
    '''
    return pd.read_csv(hits_fpath, 
                       sep=sep, 
                       encoding=encoding, 
                       index_col=index_col, 
                       dtype={'event': np.int32, 
                              'x': np.float32, 
                              'y': np.float32, 
                              'z': np.float32, 
                              'station': np.int32})


@timeit
def drop_short_tracks(mc_df, hits_df, n_points=3):
    gp_size = mc_df.groupby(['event', 'track']).size()
    # extract groups with size more than 2
    gp = gp_size[gp_size >= n_points]
    # create multiindex
    mc_df = mc_df.set_index(['event', 'track'])
    # mask dataframe
    mc_df = mc_df.loc[gp.index].reset_index()
    # after cleaning some events may be fully removed, 
    # so remove also in hits_df
    removed_events = set(hits_df.event) - set(mc_df.event)
    hits_df = hits_df[~hits_df.event.isin(removed_events)]
    return mc_df, hits_df
    

@timeit
def drop_spinning_tracks(mc_df, n_points=1):
    gp_size = mc_df.groupby(['event', 'track']).station.value_counts()
    # extract only tracks with one point per station
    gp = gp_size[gp_size == n_points]
    # create multiindex
    mc_df = mc_df.set_index(['event', 'track'])
    # mask dataframe
    idx = gp.index.droplevel('station').unique()
    return mc_df.loc[idx].reset_index()


@timeit
def label_hits(mc_df, hits_df):
    hits_with_track_id = []

    for event_id, event_data in tqdm(hits_df.groupby('event')):
        # extract event_mc_points
        event_mc_points = mc_df[mc_df.event==event_id]
        # event 663 - 9K hits
        hits_with_track_id.extend(
            match_hits_to_mc_points(
                event_id,
                event_data[['x', 'y', 'z']].values,              
                event_mc_points[['x', 'y', 'z']].values,
                event_mc_points.track.values))

    # create dataframe
    hits_with_track_id_df = pd.DataFrame(hits_with_track_id,
        columns=['event', 'x', 'y', 'z', 'track'])

    # data types conversion
    hits_with_track_id_df = hits_with_track_id_df.astype({
        'event': np.int32,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'track': np.int32})

    return hits_with_track_id_df


@timeit
def merge_mc_with_hits(mc_fpath, hits_fpath):
    print("1. Read data...")
    mc_df = read_mc_file(mc_fpath)
    hits_df = read_hits_file(hits_fpath)

    print("2. Remove tracks containing less than 3 points...")
    mc_df, hits_df = drop_short_tracks(mc_df, hits_df)

    print("3. Drop spinning tracks...")
    mc_df = drop_spinning_tracks(mc_df)

    print("4. Set labels to hits")
    hits_with_track_id_df = label_hits(mc_df, hits_df)
    
    return hits_with_track_id_df


@timeit
@plac.annotations(
    datapath=("Path to the directory with root files", "positional", None, str))
def main(datapath):
    mc_files = sorted(glob(os.path.join(datapath, 'evetest*')))
    hits_files = sorted(glob(os.path.join(datapath, 'bmndst*')))

    for mc_fpath, hits_fpath in zip(mc_files, hits_files):
        print("Files:\n\t%s\n\t%s" % (mc_fpath, hits_fpath))
        df = merge_mc_with_hits(mc_fpath, hits_fpath)
        # create name of the file to save
        save_fname = os.path.split(mc_fpath)[1]
        save_fname = save_fname.split("evetest_")[1]
        save_fname = os.path.join(datapath, save_fname)
        # save to the same location with different name
        print("Save data into `%s`" % save_fname)
        print("--- OK ---\n")
        df.to_csv(save_fname, encoding='utf-8', index=None, sep='\t')


if __name__ == "__main__":
    plac.call(main)