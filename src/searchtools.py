import model
import keras

import tensorflow as tf

from MLPtools import scale_ragged, atomic_MSE
import multiprocess as mp
from keras.losses import MeanSquaredError

from tqdm import tqdm


def build_MLP(n_neurons=10, learning_rate=0.0004, atomic_loss=True, activation="relu"):
    layers = [keras.layers.Dense(n_neurons, activation=activation),
              keras.layers.Dense(n_neurons, activation=activation)]
    if atomic_loss:
        ULM = MeanSquaredError()
        LM = None
    else:
        ULM = None
        LM = "mse"

    MLP1 = model.MLPNet(layers=layers,
                        N_features=70,
                        ragged_processing=False,
                        unitwise_loss_model=ULM
    )

    MLP1.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss=LM
    )

    return MLP1


def grid_search(X, y, grid, epochs, n_jobs):
    def test_params(config):
        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        parameters = without_keys(config, "batch_size")
        MLP = build_MLP(**parameters)
        res = MLP.fit(
            X, y,
            batch_size = config["batch_size"],
            epochs = epochs,
            verbose = 0
        )
    
        train_score = MLP.evaluate(Xtrain, y_train)
        test_score = MLP.evaluate(Xtest, y_test)
    
        return train_score, test_score, res

    with mp.Pool(n_jobs) as p:
        results_r2, train_scores_r2, test_scores_r2 = zip(*tqdm(p.map(test_params, grid)))
    return results_r2, train_scores_r2, test_scores_r2
