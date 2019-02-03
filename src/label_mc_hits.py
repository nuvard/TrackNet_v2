import pandas as pd
import numpy as np
import plac
import time

from scipy.spatial import cKDTree
from sklearn import preprocessing
from tqdm import tqdm
tqdm.pandas()


def match_hits_to_mc_points(event_hits, event_mc_df):
    # labels column
    track_ids = np.full(len(event_hits), -1, dtype='int8')
    # build KD-Tree
    #kd_tree = cKDTree(event_hits)
    # query nearest neighbours
    # distances, indices
    #dd, ii = kd_tree.query(event_mc_df[['x_in', 'y_in', 'z_in']], k=1, n_jobs=-1)
    
    # Faster than KD-Tree for very large matrices
    hits_l2 = preprocessing.normalize(event_hits, norm='l2')
    mc_points_l2 = preprocessing.normalize(event_mc_df[['x_in', 'y_in', 'z_in']])
    similarity_matrix = np.matmul(hits_l2, mc_points_l2.T)
    ii = np.argmax(similarity_matrix, axis=0)
    track_ids[ii] = event_mc_df.track_id.copy()
    return track_ids


@plac.annotations(
    mc_points_file_path=("Path to the .root file with Monte-Karlo points", "positional", None, str),
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
    print("3. Building KDTree for each event and match hits to mc_points...")
    track_ids = []
    i = 0
    for group, event_data in tqdm(hits_df.groupby('event_id')):
        # event 663 - 9K hits
        track_ids.extend(
            match_hits_to_mc_points(event_data[['x', 'y', 'z']], 
                                    mc_df[mc_df.event_id==group]))
        i += 1
    # add track_ids to hits dataframe as labels
    hits_df['track_id'] = track_ids.copy()
    print(hits_df.columns)
    print("Complete in %.2fs\n" % (time.time() - step_start_time))

    # save df to file
    print("Saving data...")
    #hits_df.to_csv('data/10K_hits.csv', encoding='utf-8', index=None)
    end_time = time.time()
    print("Elapsed: %.2f" % (end_time-start_time))


if __name__ == "__main__":
    plac.call(main)