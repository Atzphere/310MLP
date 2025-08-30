import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tqdm import tqdm

mse = keras.losses.MeanSquaredError()


def atomic_energies(y, n_atoms):
    return y / tf.cast(n_atoms, y.dtype)


def get_metrics(y_true, y_pred, n_atoms, metrics, atomic=False):
    results = []
    for metric in metrics:
        if atomic:
            res = metric(atomic_energies(y_true, n_atoms),
                         atomic_energies(y_pred, n_atoms))
        else:
            res = metric(y_true, y_pred)
        results.append(res)
        print(results)

    return results


def atomic_MSE(y_true, y_pred, n_atoms):
    return mse(atomic_energies(y_true, n_atoms), atomic_energies(y_pred, n_atoms))


def scale_ragged(feats, fit_target=None):
    stacked = np.vstack(fit_target if fit_target else feats)
    SSC = StandardScaler().fit(stacked)

    scaled_features = [SSC.transform(struct) for struct in feats]

    return scaled_features


def moving_average(x, y, window):
    if window < 1 or window % 2 == 0:
        raise ValueError("Window size must be a positive odd integer.")

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Calculate moving average using convolution
    weights = np.ones(window) / window
    y_avg = np.convolve(y, weights, mode='valid')

    # Rolling standard deviation
    y_std = np.array([
        np.std(y[i:i + window])
        for i in range(len(y) - window + 1)
    ])

    # Adjust x to align with 'valid' convolution
    x_avg = x[window // 2: -(window // 2)]

    return x_avg, y_avg, y_std


def shuffle_dataset(inputs, seed=None):
    """
    Shuffles a tuple of TensorFlow RaggedTensors and non-ragged Tensors
    along the first dimension using a shared random seed.

    Args:
        inputs: Tuple of Tensors or RaggedTensors to be shuffled.
        seed: Optional; int seed for reproducibility.

    Returns:
        Tuple of shuffled Tensors or RaggedTensors.
    """
    # Ensure inputs are a tuple
    if not isinstance(inputs, (tuple, list)):
        raise ValueError(
            "inputs must be a tuple or list of Tensors or RaggedTensors.")

    # Get the size of the first dimension
    size = tf.shape(inputs[0])[0]

    if seed is None:
        gen_seed = np.random.randint(0, 20000000)
    elif isinstance(seed, int):
        gen_seed = seed

    # Generate a shared shuffled index
    shuffled_indices = tf.random.shuffle(tf.range(size), seed=gen_seed)

    # Apply the same indices to each input
    shuffled = []
    for x in inputs:
        if isinstance(x, tf.RaggedTensor):
            shuffled.append(tf.gather(x, shuffled_indices))
        else:
            shuffled.append(tf.gather(x, shuffled_indices))

    return tuple(shuffled)


def cross_val_eval(*args, estimator, folds=1, metrics=[], atomic=False, gen_seed=None):
    gen = np.random.default_rng(seed=2)
    seeds = gen.integers(0, 2000000, size=folds)

    results = []
    X, y, n_atoms = args

    for seed in tqdm(seeds):
        X_shuf, y_shuf, n_shuf = shuffle_dataset(args, seed=gen_seed)
        y_pred = estimator.predict(X_shuf, verbose=0)
        results.append(get_metrics(
            y_shuf, y_pred, n_shuf, metrics, atomic=atomic))

    results = np.array(results)
    print(results)

    return results.mean(axis=0), results.std(axis=0)


def transform_structures(structures, lattice_vectors, transformation, variables):
    struct_series = []
    LV_series = []
    for struct0, LV0 in zip(structures, lattice_vectors):
        transformed_structs = []
        transformed_LVs = []

        for params in variables:
            result = transformation(struct0, LV0, *params)
            transformed_structs.append(result[0])
            transformed_LVs.append(result[1])

        struct_series.append(transformed_structs)
        LV_series.append(transformed_LVs)

    return struct_series, LV_series


def get_coordinates(struct):
    return struct.drop("Atom", axis="columns")
