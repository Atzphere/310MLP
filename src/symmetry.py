import numpy as np
import itertools
import pandas as pd

'''
TODO: Unit cell compensation

We are trying to generate symmetry functions that represent a 

There is a problem however: consider two different atoms A1 and B1 on opposite sides of the unit cell 1.
While A1 and B1 might interact, the interaction is likely weaker than say A1 with B-1, which is still a 


'''

def correct_closest_atoms(target_loc, neighbor_locations, COB, COB_inv):
    '''
    If a given atom is more than half a lattice a vector
    away from the target atom, that means there is an instance
    of that atom in an adjacent unit cell which is closer to the
    target. This helper function adjusts these when needed.

    target_loc: ndarray
        Row vector representing a target atom's location

    neighbor_locations ndarray:
        N x 3 array of row vectors representing the location of
        each neighboring atom.

    COB ndarray:
        Transposed change-of-basis matrix to switch to the
        basis of a structure's  mitive lattice vectors.
        Transposed to handle row vector arrays.

    COB_inv:
        matrix to switch back. provided rather than obtained from COB
        to avoid re-computation.
    '''

    # tranposed change of basis matrix to re-express distances in terms of PLVs.

    deltas = neighbor_locations - target_loc

    # convert to lattice basis
    lattice_deltas = deltas @ COB

    # True when a given atom is closer to our target in a different unit cell
    exceed_filt = np.abs(lattice_deltas) > 0.5

    # correct deltas by moving by one lattice vector in the
    # direction that minimizes distance (pick an adjacent unit cell)
    corrections = -(exceed_filt * np.sign(lattice_deltas))
    lattice_deltas += corrections

    # convert to original basis, add to target_loc
    # to get corrected positions
    corrections = lattice_deltas @ COB_inv

    corrected_locs = target_loc + corrections

    return corrected_locs


def get_radial_interaction_sets(structure, lattice_vectors):
    struct_array = structure.to_numpy()
    species = struct_array[:, 0]
    coordinates = struct_array[:, 1:].astype("float")

    pairs = list(itertools.permutations(zip(species, coordinates), 2))

    # in the case a diatomic structure, this yields 1
    window = len(structure) - 1

    radial_interaction_sets = []

    COB_matrix = np.linalg.inv(lattice_vectors)

    for i in range(len(structure)):

        interval_start = window * i
        interval_end = interval_start + window
        interactions = pairs[interval_start: interval_end]


        target_atom = interactions[0][0]
        target_loc = target_atom[1]
        neighbor_species, neighbor_locations = zip(
            *[interaction[1] for interaction in interactions])
        # neighbor species is the types (e.g. Ti, O) of the atom's neighbors
        neighbor_species = np.array(neighbor_species)
        neighbor_locations = np.array(neighbor_locations)

        corrected_locs = correct_closest_atoms(target_loc, neighbor_locations, COB_matrix, lattice_vectors)

        radial_interaction_sets.append(
            (target_atom, neighbor_species, corrected_locs))

    return radial_interaction_sets


def get_angular_interaction_sets(structure, lattice_vectors):
    struct_array = structure.to_numpy()
    species = struct_array[:, 0]
    coordinates = struct_array[:, 1:].astype("float")
    triplets = list(itertools.permutations(zip(species, coordinates), 3))
    window = (len(structure) - 1) * (len(structure) - 2)

    angular_interaction_sets = []

    COB_matrix = np.linalg.inv(lattice_vectors)

    for i in range(len(structure)):

        interval_start = window * i
        interval_end = interval_start + window
        interactions = triplets[interval_start: interval_end]

        target_atom = interactions[0][0]
        target_loc = target_atom[1]

        neighbors1, neighbors2 = zip(
            *[interaction[1:] for interaction in interactions])

        ns1, nl1 = zip(*neighbors1)
        ns2, nl2 = zip(*neighbors2)

        ns1, ns2 = np.array(ns1), np.array(ns2)

        nl1 = np.array(nl1)
        nl2 = np.array(nl2)

        nl1_cor = correct_closest_atoms(target_loc, nl1, COB_matrix, lattice_vectors)
        nl2_cor = correct_closest_atoms(target_loc, nl2, COB_matrix, lattice_vectors)

        angular_interaction_sets.append((target_atom, ns1, nl1_cor, ns2, nl2_cor))

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

    assert (len(dist[dist <= Rc]) == len(dist)), "Behler cutoff missed some entries"
    pre[dist <= Rc] = 0.5 * (np.cos(np.pi * (dist / Rc)) + 1)
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


def behler_rad(target_pos, neighbor_locs, Rs=0, eta=0):
    return np.exp(-eta * (eucl_dist(target_pos, neighbor_locs) - Rs)**2)


def behler_ang(target_pos, n1_locs, n2_locs, Rs=0, eta=0, zeta=0, lmbda=0):
    Rij = target_pos - n1_locs
    Rik = target_pos - n2_locs
    Rjk = n1_locs - n2_locs

    len_ij = length(Rij)
    len_ik = length(Rik)
    len_jk = length(Rjk)

    cos_theta = np.sum(Rij * Rik, axis=1) / (len_ij * len_ik)

    return 2**(1 - zeta) * (1 + lmbda * cos_theta)**zeta * np.exp(-eta * (len_ij**2 + len_ik**2 + len_jk**2))


def get_radial_funcs(i_set, Rc, radfunc=behler_rad, params={}):
    target = i_set[0]
    target_pos = target[1]

    ns = i_set[1]
    nl = i_set[2]

    fc = behler_cutoff(target_pos, nl, Rc)

    unique_species = set(ns)

    if not (len(params.keys()) >= len(unique_species)):
        raise ValueError(
            "Radial parameters were not specified for all interaction types in crystal")

    output_features = []
    for species in params.keys():
        filt = ns == species
        atom_locs = nl[filt]
        func_params = params[species]
        # supports multiple symmetry functions per species

        for paramset in func_params:
            terms = radfunc(target_pos, atom_locs, **paramset) * fc[filt]
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

    # V2: this actually causes undercounting? I believe the actual summation
    # should include the degeneracy; we do not want to sort the pairs.

    # pairs = [tuple(sorted(pair)) for pair in np.vstack([ns1, ns2]).T]

    pairs = [tuple(pair) for pair in np.vstack([ns1, ns2]).T]

    unique_species = set(pairs)
    pairs = np.array(pairs)

    # this deals with ordering; Ti-O is considered the same as O-Ti
    # sorting the input params puts them in a canonical order
    # V2: we actually want to generate get the complementary reversed pair as well

    expanded_params = params.copy()

    for key in params.keys():
        rev = tuple(reversed(key))
        if rev not in expanded_params.keys():
            expanded_params.update({rev: params[key]})

    cleaned_param_keys = expanded_params.keys()

    # cleaned_param_keys = set([tuple(sorted(pair)) for pair in params.keys()])

    if not unique_species.issubset(expanded_params):
        # every observed interaction type needs parameters
        raise ValueError(
            "Angular parameters were not specified for all interaction types in crystal")

    output_features = []

    for species, original in zip(cleaned_param_keys, expanded_params):
        filt = np.all(pairs == species, axis=-1)

        al1 = nl1[filt]
        al2 = nl2[filt]
        func_params = expanded_params[original]
        # supports multiple symmetry functions per species

        for paramset in func_params:
            terms = angfunc(target_pos, al1, al2, **paramset) * fc[filt]
            output_features.append(np.sum(terms))

    return np.array(output_features)


def get_features(structure, lattice_vectors, radfunc, angfunc, Rc, params_rad, params_ang):
    '''
    Generates symmetry function features for a particular structure file.
    '''

    # get cutoff-culled sets of interactions for every atom in the structure
    radial_interaction_sets = [cull_radial_interaction_set(
        iset, Rc=Rc) for iset in get_radial_interaction_sets(structure, lattice_vectors)]

    angular_interaction_sets = [cull_angular_interaction_set(
        iset, Rc=Rc) for iset in get_angular_interaction_sets(structure, lattice_vectors)]

    atoms = zip(radial_interaction_sets, angular_interaction_sets)

    features = []

    for radial, angular in atoms:
        radial_features = get_radial_funcs(
            radial, Rc, radfunc=radfunc, params=params_rad)
        angular_features = get_angular_funcs(
            angular, Rc, angfunc=angfunc, params=params_ang)

        feature_vector = np.concatenate([radial_features, angular_features])
        features.append(feature_vector)

    return np.array(features)


def behler_features(structure, LV, Rc, params_rad, params_ang):
    '''
    Generates symmetry function features based off of the method detailed
    in Behler and Parrinello 2007 for a given structure.
    '''

    return get_features(structure, LV, radfunc=behler_rad, angfunc=behler_ang,
                        Rc=Rc, params_rad=params_rad, params_ang=params_ang)


# class MLPParams(object):
#     def __init__(self, Rc, rad_params, ang_params, global_Rs=None):
#         self.Rc = Rc
#         if global_Rs:
#             self.global_Rs = global_Rs
#         self.rad_species = list(rad_params.keys())
#         self.ang_species = list(ang_params.keys())

# class SpeciesCollection(object):


# class Species(object):
#     '''
#     overloading specifications:

#     Non-ordered but NOT a set:
#     Symbol([A, B]) == Symbol([B, A])
#     Symbol([A, A]) != Symbol([A])

#     Commutation under multiplication:
#     A * B == B * A == Symbol([A, B])

#     Commutation under addition (with an auxiliary object)
#     A + B == B + A == SymbolCollection([A, B])

#     Distributive property
#     A * (A + B) == SymbolCollection([A * A, A * B]) == SymbolCollection(Symbol([A, A]), Symbol([A, B]))

#     Associativity:
#     A * B * C == (A * B) * C == A * (B * C)

#     More distributive property:

#     (A + B) * (C + D) == SymbolCollection([A * C, A * D, B * C, B * D])


#     attributes:
#     constituent atoms (as a set?)
#     order: how many atoms are in the
#     '''

#     def __init__(self, species):
#         if isinstance(species, Iterable):
#             self.species = Counter(species)
#         else:
#             self.species = Counter([species])

#     def __mul__(self, other):
#         return Species([self, other])

#     def __add__(self, other):
