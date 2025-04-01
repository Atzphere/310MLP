import pandas as pd
import dirtools as dt
from tqdm import tqdm
import numpy as np
import multiprocess as mp
import os
import symmetry
import pathlib
import h5py
# from multiprocess import Pool

def load_dset(dir_path, name, features_path=None, labels_path=None):
    '''
    the latter 2 kwargs are overrides to specify explicit paths for both
    '''
    if labels_path and features_path:
        lpath = labels_path
        fpath = features_path
    elif (labels_path and features_path) and labels_path != features_path:
        raise ValueError("features_path and labels_path must both be supplied or absent")
    else:
        dir_path = pathlib.Path(dir_path)
        fpath = dir_path / f"{name}_features.h5"
        lpath = dir_path / f"{name}_labels.h5"
    with h5py.File(fpath, "r") as f:
        features = [f[f"array_{i}"][:] for i in range(len(f))]

    with h5py.File(lpath, "r") as f:
        labels = [f[f"array_{i}"][:] for i in range(len(f))]

    return features, labels

def load_crystal(fname):
    '''
    Reads an .xsf file and returns a 2D array of the
    x, y, and z positions of the relevant atoms, as well as
    the total DFT-calculated energy of the configuration.
    '''

    df = pd.read_csv(fname, skiprows=9, sep="\s+", names=["Atom", "x", "y", "z"], header=None, usecols=[0, 1, 2, 3])

    with open(fname, "r") as f:
        energy = float(f.readline().split(" ")[4])

    return df, energy


def get_dataset(dirname, ext=".xsf", halt=-1):
    print(fr"Loading structure data: {os.path.basename(dirname)}")
    files = list(dt.get_files(dirname, fullpath=True, file_filter=[ext]))[:halt]

    with mp.Pool(mp.cpu_count()) as p:
        configurations, energies = zip(*tqdm(p.imap(load_crystal, files), total=len(files)))
    print("Loaded.")

    return configurations, np.array(energies).reshape(-1, 1)


def build_features(structs, Rc, params_rad, params_ang):
    print(fr"Building features from structure data")

    def process_structure(struct):
        return symmetry.behler_features(struct, Rc=Rc, params_rad=params_rad, params_ang=params_ang)

    with mp.Pool(mp.cpu_count()) as p:
        features = list(tqdm(p.imap(process_structure, structs), total=len(structs)))
    print("Done.")

    return features
