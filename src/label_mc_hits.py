import pandas as pd
import numpy as np
import plac
import time

from collections import Counter
from tqdm import tqdm
tqdm.pandas()

from fast_knn import knn


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
    # Faster than KD-Tree for very large matrices
    # reshape, because method returns matrix
    ii = knn(event_hits, event_mc_points).reshape(-1)
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


@plac.annotations(
    mc_points_file_path=("Path to the .root file with Monte-Carlo points", "positional", None, str),
    hits_file_path=("Path to the .root file with reconstructed hits", "positional", None, str))
def main(mc_points_file_path, hits_file_path):
    start_time = time.time()
    print("1. Reading data...")
    mc_df = pd.read_csv(mc_points_file_path, encoding='utf-8')
    hits_df = pd.read_csv(hits_file_path, encoding='utf-8')
    print("Complete in %.2fs\n" % (time.time() - start_time))

    # clean data from small tracks
    step_start_time = time.time()
    print("2. Removing tracks containing less than 3 points...")
    mc_df = mc_df.groupby(['event_id', 'track_id']).filter(lambda x: len(x) > 2)
    # after cleaning some event may be fully removed, so remove also in hits_df
    # {1347, 3155, 3799, 8025, 8350, 8889, 8956}
    removed_events = set(hits_df.event_id) - set(mc_df.event_id)
    hits_df = hits_df[~hits_df.event_id.isin(removed_events)]
    print("Complete in %.2fs\n" % (time.time() - step_start_time))

    # build
    step_start_time = time.time()
    print("3. Matching hits to mc_points...")
    hits_with_track_id = []
    i = 0
    for event_id, event_data in tqdm(hits_df.groupby('event_id')):
        # extract event_mc_points
        event_mc_points = mc_df[mc_df.event_id==event_id]
        # event 663 - 9K hits
        hits_with_track_id.extend(
            match_hits_to_mc_points(
                event_id,
                event_data[['x', 'y', 'z']].values,              
                event_mc_points[['x_in', 'y_in', 'z_in']].values,
                event_mc_points.track_id.values))
        i += 1
    # remove from memory existing dataframes
    del hits_df, mc_df
    # create new dataframe
    hits_with_track_id_df = pd.DataFrame(
        data=hits_with_track_id,
        columns=['event_id', 'x', 'y', 'z', 'track_id'])
    hits_with_track_id_df = hits_with_track_id_df.astype({
        'event_id': np.int32,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'track_id': np.int32})
    print("Complete in %.2fs\n" % (time.time() - step_start_time))

    # save df to file
    print("Saving data...")
    hits_with_track_id_df.to_csv('data/10K_hits.csv', encoding='utf-8', index=None)
    end_time = time.time()
    print("Elapsed: %.2f" % (end_time-start_time))


if __name__ == "__main__":
    plac.call(main)