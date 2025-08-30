import pandas as pd
import dirtools as dt
from tqdm import tqdm
import numpy as np
import multiprocess as mp
import os
import symmetry
import pathlib
import h5py
import re
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


def parse_line(line):
    '''
    parses a line of lattice vector data
    '''
    return list(map(float, line.strip().split()))


def load_crystal(fname):
    '''
    Reads an .xsf file and returns a 2D array of the
    x, y, and z positions of the relevant atoms, as well as
    the total DFT-calculated energy of the configuration.

    Also reads the included lattice vector information for nearest-neighbor corrections.

    As well as the counts of each atom type; these are later used alongside single-atom energies
    to extract cohesive energy data from the raw DFT total structure energies

    returns a tuple of outputs:

    configuration: a dataframe with columns corresponding to the type and position (cartesian)
    of every atom in the structure.

    energy: a float; the total structure energy.

    lattice vectors: an array of the structures lattice row vectors

    count: a dictionary with items {atom_name: int}; the number of each type of atom in the structure
    '''

    struct_df = pd.read_csv(fname, skiprows=9, sep=r"\s+", names=["Atom", "x", "y", "z"], header=None, usecols=[0, 1, 2, 3])

    count = dict(struct_df["Atom"].value_counts())

    with open(fname, "r") as f:
        energy = float(f.readline().split(" ")[4])
        lattice_vectors = []
        for i in range(3):
            f.readline()
        for i in range(3):
            LV = parse_line(f.readline())
            lattice_vectors.append(tuple(LV))

    return struct_df, energy, np.array(lattice_vectors), count


def get_dataset(dirname, ext=".xsf", halt=-1):
    print(fr"Loading structure data: {os.path.basename(dirname)}")
    files = list(dt.get_files(dirname, fullpath=True, file_filter=[ext]))[:halt]

    with mp.Pool(mp.cpu_count()) as p:
        configurations, energies, LVs, counts_list, = zip(*tqdm(p.map(load_crystal, files), total=len(files)))

    # consolidate counts into a df
    counts = pd.DataFrame(counts_list).fillna(0)
    print("Loaded.")

    return configurations, np.array(energies).reshape(-1, 1), LVs, counts


def build_features(structs, LVs, Rc, params_rad, params_ang, n_jobs=-1):
    print(fr"Building features from structure data")

    def process_structure(struct, LV):
        return symmetry.behler_features(struct, LV, Rc=Rc, params_rad=params_rad, params_ang=params_ang)

    pool_size = n_jobs if n_jobs != -1 else mp.cpu_count()

    with mp.Pool(pool_size) as p:
        features = list(tqdm(p.imap(lambda zoop: process_structure(*zoop), zip(structs, LVs)), total=len(structs)))

    print("Done.")

    return features


def get_cohesive_energies(total_energies, counts, atomic_energies):
    '''
    Gets the total cohesive energies from a set of structures given their
    total DFT structure energy (i.e. the raw label energies),
    composition, and individual-atom DFT energies.
    '''

    if not atomic_energies.keys() >= set(counts.keys()):
        raise ValueError("Atomic energies must be supplied for all species in a structure.")

    contributions = counts.mul(atomic_energies, axis="columns", fill_value=0).sum(axis=1).values.reshape(-1, 1)

    return total_energies - contributions
