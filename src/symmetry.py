import numpy as np
import itertools
import pandas as pd


def get_radial_interaction_sets(structure):
    struct_array = structure.to_numpy()
    species = struct_array[:, 0]
    coordinates = struct_array[:, 1:].astype("float")
    pairs = list(itertools.permutations(zip(species, coordinates), 2))
    window = len(structure) - 1

    radial_interaction_sets = []

    for i in range(len(structure)):
        neighbor_species = []
        neighbors = []

        interval_start = window * i
        interval_end = interval_start + window
        interactions = pairs[interval_start: interval_end]

        target_atom = interactions[0][0]
        neighbor_species, neighbor_locations = zip(
            *[interaction[1] for interaction in interactions])

        neighbor_species = np.array(neighbor_species)
        neighbor_locations = np.array(neighbor_locations)

        radial_interaction_sets.append(
            (target_atom, neighbor_species, neighbor_locations))

    return radial_interaction_sets


def get_angular_interaction_sets(structure):
    struct_array = structure.to_numpy()
    species = struct_array[:, 0]
    coordinates = struct_array[:, 1:].astype("float")
    triplets = list(itertools.permutations(zip(species, coordinates), 3))
    window = (len(structure) - 1) * (len(structure) - 2)

    angular_interaction_sets = []

    for i in range(len(structure)):
        neighbor_species = []
        neighbors = []

        interval_start = window * i
        interval_end = interval_start + window
        interactions = triplets[interval_start: interval_end]

        target_atom = interactions[0][0]

        neighbors1, neighbors2 = zip(
            *[interaction[1:] for interaction in interactions])

        ns1, nl1 = zip(*neighbors1)
        ns2, nl2 = zip(*neighbors2)

        ns1, ns2 = np.array(ns1), np.array(ns2)

        nl1 = np.array(nl1)
        nl2 = np.array(nl2)

        angular_interaction_sets.append((target_atom, ns1, nl1, ns2, nl2))

    return angular_interaction_sets


def eucl_dist(X1, X2):
    return np.linalg.norm(X1 - X2, axis=-1)


def length(X):
    return np.linalg.norm(X, axis=-1)


def out_of_range(X1, X2, Rc):
    return eucl_dist(X1, X2) > Rc


def behler_cutoff(X1, X2, Rc):
    dist = eucl_dist(X1, X2)
    pre = np.zeros_like(dist)
    pre[dist <= Rc] = 0.5 * (np.cos(np.pi * dist[dist <= Rc] / Rc) + 1)
    return pre


def cull_radial_interaction_set(i_set, Rc):
    target = i_set[0]
    target_pos = target[1]

    neighbor_species = i_set[1]
    neighbors = i_set[2]

    within_range = ~out_of_range(target_pos, neighbors, Rc)

    return (target, neighbor_species[within_range], neighbors[within_range])


def cull_angular_interaction_set(i_set, Rc):
    target = i_set[0]
    target_pos = target[1]

    ns1 = i_set[1]
    nl1 = i_set[2]

    ns2 = i_set[3]
    nl2 = i_set[4]

    WR1 = ~out_of_range(target_pos, nl1, Rc)
    WR2 = ~out_of_range(target_pos, nl2, Rc)
    WR3 = ~out_of_range(nl1, nl2, Rc)

    valid = WR1 & WR2 & WR3

    return (target, ns1[valid], nl1[valid], ns2[valid], nl2[valid])


def get_radial_fingerprint(interaction, params):
    pass


def get_fingerprint_vector():
    pass


def behler_rad(target_pos, neighbor_locs, Rs, eta):
    return np.exp(-eta * (eucl_dist(target_pos, neighbor_locs) - Rs)**2)


def behler_ang(target_pos, n1_locs, n2_locs, Rs, eta, zeta, lmbda):
    Rij = target_pos - n1_locs
    Rik = target_pos - n2_locs
    Rjk = n1_locs - n2_locs

    len_ij = length(Rij)
    len_ik = length(Rik)
    len_jk = length(Rjk)

    theta = np.sum(Rij * Rik, axis=1) / (len_ij * len_ik)

    return 2**(1 - zeta) * (1 + lmbda * np.cos(theta))**zeta * np.exp(-eta * (len_ij**2 + len_ik**2 + len_jk**2))


def get_radial_funcs(i_set, Rc, radfunc=behler_rad, params={}):
    target = i_set[0]
    target_pos = target[1]

    ns = i_set[1]
    nl = i_set[2]

    fc = behler_cutoff(target_pos, nl, Rc)

    unique_species = set(ns)

    if not (params.keys() >= unique_species):
        raise ValueError(
            "Parameters were not specified for all interaction types in crystal")

    output_features = []
    for species in params.keys():
        filt = ns == species
        atom_locs = nl[filt]
        func_params = params[species]
        # supports multiple symmetry functions per species

        for paramset in func_params:
            terms = radfunc(target_pos, atom_locs, *paramset) * fc[filt]
            output_features.append(np.sum(terms))

    return np.array(output_features)


def get_angular_funcs(i_set, Rc, angfunc=behler_ang, params={}):
    target = i_set[0]
    target_pos = target[1]

    ns1 = i_set[1]
    nl1 = i_set[2]

    ns2 = i_set[3]
    nl2 = i_set[4]

    fcij = behler_cutoff(target_pos, nl1, Rc)
    fcik = behler_cutoff(target_pos, nl2, Rc)
    fcjk = behler_cutoff(nl1, nl2, Rc)

    fc = fcij * fcik * fcjk

    pairs = [tuple(sorted(pair)) for pair in np.vstack([ns1, ns2]).T]
    unique_species = set(pairs)
    pairs = np.array(pairs)

    cleaned_param_keys = set([tuple(sorted(pair)) for pair in params.keys()])

    if not (cleaned_param_keys >= unique_species):
        raise ValueError(
            "Parameters were not specified for all interaction types in crystal")

    output_features = []
    for species, original in zip(cleaned_param_keys, params.keys()):
        filt = np.all(pairs == species, axis=-1)

        al1 = nl1[filt]
        al2 = nl2[filt]
        func_params = params[original]
        # supports multiple symmetry functions per species

        for paramset in func_params:
            terms = angfunc(target_pos, al1, al2, *paramset) * fc[filt]
            output_features.append(np.sum(terms))

    return np.array(output_features)


def get_features(structure, radfunc, angfunc, Rc, params_rad, params_ang):
    '''
    Generates symmetry function features for a particular structure file.
    '''

    # get cutoff-culled sets of interactions for every atom in the structure
    radial_interaction_sets = [cull_radial_interaction_set(iset, Rc=Rc) for iset in get_radial_interaction_sets(structure)]

    angular_interaction_sets = [cull_angular_interaction_set(iset, Rc=Rc) for iset in get_angular_interaction_sets(structure)]

    atoms = zip(radial_interaction_sets, angular_interaction_sets)

    features = []

    for radial, angular in atoms:
        radial_features = get_radial_funcs(
            radial, Rc, radfunc=radfunc, params=params_rad)
        angular_features = get_angular_funcs(
            angular, Rc, angfunc=angfunc, params=params_ang)

        feature_vector = np.concatenate([radial_features, angular_features])
        features.append(feature_vector)

    return features


def behler_features(structure, Rc, params_rad, params_ang):
    '''
    Generates symmetry function features based off of the method detailed
    in Behler and Parrinello 2007 for a given structure.
    '''

    return get_features(structure, radfunc=behler_rad, angfunc=behler_ang,
                        Rc=Rc, params_rad=params_rad, params_ang=params_ang)
