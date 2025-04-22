import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

mse = keras.losses.MeanSquaredError()


def atomic_energies(y, n_atoms):
    return y / n_atoms


def atomic_MSE(y_true, y_pred, n_atoms):
    return mse(atomic_energies(y_true, n_atoms), atomic_energies(y_pred, n_atoms))


def scale_ragged(features):
    stacked = np.vstack(features)
    SSC = StandardScaler().fit(stacked)
    scaled_features = [SSC.transform(struct) for struct in features]

    return scaled_features
