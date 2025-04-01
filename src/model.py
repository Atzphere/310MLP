import tensorflow as tf
import tf_keras as keras
import tqdm.keras


# def map_fn2(fn, elems, fn_output_signature):
#     batch_size = tf.shape(tf.nest.flatten(elems)[0])[0]
#     arr = tf.TensorArray(
#         fn_output_signature.dtype, size=batch_size, element_shape=fn_output_signature.shape)
#     for i in tf.range(batch_size):
#         arr = arr.write(i, fn(tf.nest.map_structure(lambda x: x[i], elems)))
#     return arr.stack()


class ReduceRegressor(keras.Model):
    '''
    Dynamic-topography neural network which takes a ragged input tensor
    of shape (N_instances, None, N_features), passes N_subinstances (the ragged dimension)
    to a subnet, and then reduces the result to a tensor with label-like dimension.
    The same subnet is used for each individual subinstance forward pass.
    '''

    def __init__(self, subnet, N_features, reduction_func, reduction_params=None, ragged_processing=True):
        '''
        pad_size : int or None, default None
            Whether or not to pad ragged inputs rather than processing them as-is.
            Padding improves model compatibility, namely you should be able to run
            the model in Keras 3, and make use of GPU resources.

            Setting this to -1 will make the model automatically pad to size of the
            largest instance in the ragged dimension in the training set.

            Note that this will inherently constraint the size of any inference features
            to pad_size.

        ragged_processing : bool, default True
            Whether or not to handle the ragged input tensor as-is or to
            dynamically pad it on a per-batch basis. The latter allows for
            GPU computation as well as compatibility with Keras 3, but may
            be less understandable.

        '''
        super().__init__()
        self.subnet = subnet
        self.reduction_func = reduction_func
        self.reduction_params = reduction_params
        self.ragged_processing = ragged_processing
        self.N_features = N_features

    def get_subnet(self):
        return self.subnet

    def call(self, inputs, training=False):

        def _process_subinstance(struct):
            return self.subnet(struct, training=training)

        if self.ragged_processing:
            pairwise_contribs = tf.map_fn(
                _process_subinstance, inputs, fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0))

            return self.reduction_func(pairwise_contribs, **self.reduction_params)

    def fit(self, x=None, y=None, **kwargs):
        '''
        overriden fit with fancy progress bar and pre-processing
        to handle ragged tensors nicely
        '''
        # intercept kwargs to construct a tf.dataset object if needed
        # do this for both training, validation sets (if present)

        batch_size = kwargs.pop("batch_size", 32)

        def _make_dataset(X, y, sample_weight=None):
            if isinstance(X, tf.data.Dataset):
                if X.element_spec.shape[0] not in (2, 3):
                    raise ValueError(
                        "A Dataset supplied to fit() should return a tuple of either (inputs, targets) or (inputs, targets, sample_weights).")
                else:
                    dataset = X
            else:
                # a little horrible but it should work...?
                def __gen():
                    if sample_weight:
                        for features, label, weights in zip(X, y, sample_weight):
                            yield features, label, weights
                    else:
                        for features, label in zip(X, y):
                            yield features, label

                dataset = tf.data.Dataset.from_generator(
                    generator=__gen,
                    output_signature=(tf.TensorSpec(shape=(None, self.N_features), dtype=tf.float32),
                                      tf.TensorSpec(shape=(1,), dtype=tf.float32)))

            # handle ragged padding depending on initialization setting

            if not self.ragged_processing:
                processed = dataset.padded_batch(
                    batch_size=batch_size, padded_shapes=(None, self.N_features))
            else:
                processed = dataset.ragged_batch(batch_size=batch_size)
            print(dataset)
            print(processed.take(1))
            return processed

        X_train = x
        y_train = y
        sample_weight_train = kwargs.pop("sample_weight", None)
        train_dataset = _make_dataset(X_train, y_train, sample_weight_train)

        val_dataset = kwargs.pop("validation_data", None)
        if val_dataset:
            val_dataset = _make_dataset(*val_dataset)

        if "callbacks" not in kwargs or kwargs["callbacks"] is None:
            kwargs["callbacks"] = [tqdm.keras.TqdmCallback()]
        else:
            kwargs["callbacks"].append(tqdm.keras.TqdmCallback())

        return super().fit(x=train_dataset, validation_data=val_dataset, **kwargs)


class MLPNet(ReduceRegressor):
    def __init__(self, layers, N_features):
        subnet = keras.Sequential(layers=[
            keras.Input(shape=(N_features,))]  # input layer takes in n_inputs number of symmetry function features
            + layers
            + [keras.layers.Dense(1, activation="linear")])
        super().__init__(subnet=subnet,
                         N_features=N_features,
                         reduction_func=tf.reduce_sum,
                         reduction_params={"axis": 1})
