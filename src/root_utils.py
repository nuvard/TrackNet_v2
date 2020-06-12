import os
import shutil
import numpy as np
import pandas as pd

from datetime import datetime
from ROOT import TFile
from glob import glob


def mc2pandas(tree, attr_name):
    '''Takes root tree's data and extracts all 
    monte-carlo (mc) points

    # Arguments
        tree: ROOT.TTree
        attr_name: string, name of the tree with mc points

    # Returns
        pandas.DataFrame
    '''
    MCs = []

    print("File processing...")
    for event_id, e in enumerate(tree):
        for mc in getattr(e, attr_name):
            MCs.append([event_id,
                        mc.GetTrackID(), 
                        mc.GetXIn(), 
                        mc.GetYIn(), 
                        mc.GetZIn(), 
                        mc.GetXOut(), 
                        mc.GetYOut(), 
                        mc.GetZOut(), 
                        mc.GetStation()])
    # dataframe columns
    columns = ['event', 'track', 'x_in', 'y_in', 'z_in', 
               'x_out', 'y_out', 'z_out', 'station']
    print("Complete!")
    # create DataFrame and set columns types
    df = pd.DataFrame(data=MCs, columns=columns) \
           .astype({'event': np.int32, 
                    'track': np.int32,
                    'x_in': np.float32,
                    'y_in': np.float32,
                    'z_in': np.float32,
                    'x_out': np.float32,
                    'y_out': np.float32,
                    'z_out': np.float32,
                    'station': np.int32})
    return df


def hits2pandas(tree, attr_name):
    '''Takes root tree's data and extracts all 
    hit points

    # Arguments
        tree: ROOT.TTree
        attr_name: string, name of the tree with hits

    # Returns
        pandas.DataFrame
    '''
    hits = []

    print("File processing...")
    for event_id, e in enumerate(tree):
        for hit in getattr(e, attr_name):
            hits.append([event_id,
                         hit.GetX(),  
                         hit.GetY(), 
                         hit.GetZ(), 
                         hit.GetStation()])
    print("Complete!")
    # dataframe columns
    columns = ['event', 'x', 'y', 'z', 'station']

    # create DataFrame and set columns types
    df = pd.DataFrame(data=hits, columns=columns) \
           .astype({'event': np.int32, 
                    'x': np.float32,
                    'y': np.float32,
                    'z': np.float32,
                    'station': np.int32})
    return df


def root2pandas(fname, 
                tree_name='cbmsim', 
                hit_obj_name='BmnGemStripHit', 
                mc_obj_name='StsPoint'):
    '''Reads root file 'fname', and converts its
    contents to DataFrame

    # Arguments
        fname: string, name of the file with path to it
        tree_name: string, name of the tree with data
        hit_obj_name: string, name of the tree with hits
        mc_obj_name: string, name of the tree with Monte-Carlo points

    # Returns
        pandas.DataFrame
    ''' 
    print("Read file '%s'" % fname)
    f = TFile(fname)
    # read event tree
    tree = f.Get(tree_name)

    # if file with Monte-Carlo points
    if tree.GetBranch(mc_obj_name):
        return mc2pandas(tree, mc_obj_name)

    # if file with hits
    if tree.GetBranch(hit_obj_name):
        return hits2pandas(tree, hit_obj_name)

    # else return None
    return None


def get_new_fname(fname, path_prefix=None):
    '''Creates a new input fname using the original
    replacing the extension to tsv

    # Arguments
        fname: string, original file name or path to file
        path_prefix: string, path for the new file

    # Returns
        string, new file's name
    '''
    fpath, fname = os.path.split(fname)
    fname, ext = os.path.splitext(fname)
    fname = '.'.join([fname, 'tsv'])
    
    if path_prefix is None:
        # add old prefix if exists
        return os.path.join(fpath, fname)

    # add path prefix to filename
    return os.path.join(path_prefix, fname)


def create_new_folder(path, path_postfix, rmdir_old=True):
    '''Creates a new folder using the input path by adding
    to it a `path_postfix`. 

    If the directory with the choosed name exists 
    removes it or not depending on `rmdir_old` parameter

    # Arguments
        path: string, path to the original directory
        path_postfix: string, will be added to original to create new name
        rmdir_old: boolean, whether or not to remove old dir

    # Returns 
        string, name of the created folder
    '''
    dir_to_save = '_'.join([path, path_postfix])

    if not os.path.isdir(dir_to_save):
        os.mkdir(dir_to_save)
        print("Folder '%s' was created" % dir_to_save)
        return dir_to_save

    if rmdir_old:
        print("Folder with such name exists, remove it")
        shutil.rmtree(dir_to_save)

    if not rmdir_old:
        datetime_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_to_save = '_'.join([dir_to_save, datetime_now])

    os.mkdir(dir_to_save)
    print("Folder '%s' was created" % dir_to_save)
    return dir_to_save
    
    

def root2tsv(path):
    '''Takes as input path argument and checks 
    either it is a file or directory. 
    
    If it is a file extracts data into dataframe and 
    saves this dataframe in the same location with 
    the same name, but different extension.

    If it is a directory creates a new directory in the
    same location with different name and processes all
    contents of the input directory to save them into 
    a new recently created folder.

    # Arguments
        path: string, path to the file or directory with files
    '''
    # if file save to the current dir
    if os.path.isfile(path):
        df = root2pandas(path)
        fname = get_new_fname(path)
        df.to_csv(fname, encoding='utf-8', sep='\t')
        return

    if not os.path.isdir(path):
        raise FileNotFoundError("Path doesn't exist")

    # if directory create another dir in the same location
    print("Path argument is a directory. " 
          "Creating another folder in the same path")
    dir_to_save = create_new_folder(path, 'tsv')

    #process each file and save to dir
    for fpath in glob(os.path.join(path, '*.root')):
        df = root2pandas(fpath)
        fname = get_new_fname(fpath, path_prefix=dir_to_save)
        df.to_csv(fname, encoding='utf-8', sep='\t')