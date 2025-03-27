import tensorflow as tf
import tf_keras as keras
import tqdm.keras


class ReduceRegressor(keras.Model):
    '''
    Dynamic-topography neural network which takes a ragged input tensor
    of shape (N_instances, None, N_features), passes N_subinstances (the ragged dimension)
    to a subnet, and then reduces the result to a tensor with label-like dimension.
    The same subnet is used for each individual subinstance forward pass.
    '''

    def __init__(self, subnet, reduction_func, reduction_params=None):
        super().__init__()
        self.subnet = subnet
        self.reduction_func = reduction_func
        self.reduction_params = reduction_params

    def get_subnet(self):
        return self.subnet

    def call(self, inputs, training=False):

        def _process_subinstance(struct):
            return self.subnet(struct, training=training)

        pairwise_contribs = tf.map_fn(
            _process_subinstance, inputs, fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0))

        return self.reduction_func(pairwise_contribs, **self.reduction_params)

        def fit(self, *args, **kwargs):
            '''
            Boilerplate fit with fancy progress bar.
            '''
            if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                kwargs["callbacks"] = [tqdm.keras.TqdmCallback()]
            else:
                kwargs["callbacks"].append(tqdm.keras.TqdmCallback())
            return super().fit(*args, **kwargs)


class MLPNet(ReduceRegressor):
    def __init__(self, layers, n_syms):
        subnet = keras.Sequential(layers=[
            keras.Input(shape=(n_syms,))]  # input layer takes in n_inputs number of symmetry function features
            + layers
            + [keras.layers.Dense(1, activation="linear")])
        super().__init__(subnet=subnet, reduction_func=tf.reduce_sum,
                         reduction_params={"axis": 1})
