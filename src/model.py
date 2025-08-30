import tensorflow as tf
# import tf_keras as keras
import keras
import tqdm.keras
import numpy as np
from tqdm.auto import tqdm as tqdm_auto

from keras.losses import MeanSquaredError


def chunksum(array, window):
    '''
    Sums an array in adjacent non-overlapping windows of a predesignated size.

    For an input array of size N,
    The resulting output array will have shape (ceil(N / window),)

    The final "chunksum" might not be summed over the correct amount of entries
    depending on output.

    '''
    length = len(array)
    needed = window - (length % window)
    finlength = length + needed
    new = np.concatenate((array, np.zeros(needed)))

    return new.reshape(finlength // window, window) @ np.ones(window)


def atomic_loss(y_true, y_pred, n_atoms, model=MeanSquaredError()):
    '''
    Used by the MLPNet model to get loss per atom during training.
    '''

    float_n = tf.cast(n_atoms, tf.float32)
    errors = model(tf.math.truediv(y_true, float_n),
                   tf.math.truediv(y_pred, float_n))

    return errors


class ReduceRegressor(keras.Model):
    '''
    Dynamic-topography neural network which takes a ragged input tensor
    of shape (N_instances, None, N_features), passes N_subinstances (the ragged dimension)
    to a subnet, and then reduces the result to a tensor with label-like dimension.
    The same subnet is used for each individual subinstance forward pass.
    '''

    def __init__(self, subnet, N_features, reduction_func, reduction_params=None, ragged_processing=True, unitwise_loss_model=None):
        '''
        subnet : keras.Model
            The subnet to be evaluated and fitted on subinstances.
            The output of the forward pass of this subnet will be reduced via reduction_func
            to produce the final output shape of the ReduceRegressor.

        N_features : int
            The number of features that each substance has. Provided explicitly here
            because it saves me from having to completely override some class methods

        reduction_func : callable; Tensor[Tensor] -> Tensor
            The reduction function. This should be able to take an arbitrary-length list-like tensor of
            output tensors from the subnet, then reduce them into an output tensor with the same shape
            as your label data.

        reduction_params : dict or None, default None
            Optional list of parameters to be passed to reduction_func

        ragged_processing : bool, default True
            Whether or not to handle the ragged input tensor as-is or to
            dynamically pad it on a per-batch basis. The latter allows for
            GPU computation as well as compatibility with Keras 3, but may
            be less understandable.

        unitwise_loss_model : callable(Tensor) -> float; or None, default None
            Function to evaluate loss on a per-subinstance basis. If None, the
            loss function supplied through ReduceRegressor.compile() will be used
            instead. Providing a loss function here 


        '''
        super().__init__()
        self.subnet = subnet
        self.reduction_func = reduction_func
        self.reduction_params = reduction_params
        self.ragged_processing = ragged_processing
        self.N_features = N_features

        self.unitwise_loss_model = unitwise_loss_model

    def get_subnet(self):
        return self.subnet

    def call(self, X, training=False):
        '''

        X is a batch (list-like) of tuples: (inputs, masks, sequence_lengths)
        of length batch_size.

        "inputs" is a batch of instances with shape:
        (batch_size, None (the ragged lengths M), N_features)

        and "masks" is a batch of masks with shape:
        (batch_size, None (the ragged lengths M))

        sequence_lengths: (batch_size,) give us dimensional info that
        lets us recover our original batch structure after
        vector-optimized computing. 

        if batch padding is being used, the middle dimension
        will not be ragged but rather a fixed padding size.
        since masking is only used in the case of padding,

        "masks" should effectively only ever take the shape:
        (batch_size, M (pad size))

        '''

        # map wrapper for our subnet
        def _process_subinstance(struct):
            return self.subnet(struct, training=training)

        # first, unpack inputs into features, masks, spans:

        inputs, masks, sequence_lengths, *rest = X

        # handle y_true if supplied for error estimation
        truth_supplied = (len(rest) > 0) and tf.is_tensor(rest[0])

        y_true = rest[0] if truth_supplied else None

        # forward pass in ragged mode
        # since the array is not padded, we can simply map the subnet call over all instances.

        if self.ragged_processing:
            pairwise_contribs = tf.map_fn(
                _process_subinstance, inputs, fn_output_signature=tf.RaggedTensorSpec(ragged_rank=0))

        # forward pass in padded batch mode

        else:
            # let M_max = (size of biggest instance in M; i.e. the padded batch size)
            M_max = tf.math.reduce_max(sequence_lengths)
            batch_size = tf.shape(inputs)[0]

            # we stack all instances into one 2D tensor
            stacked_inputs = tf.reshape(inputs, (-1, self.N_features))
            # resulting shape: (batch_size * M_max, N_features)

            # we can avoid iteration and directly use the vectorized subnet.call()
            raw_contribs = _process_subinstance(stacked_inputs)
            # the output should be 1D: (batch_size * M_max,)

            # assert(raw_contribs.shape[0] % M_max == 0,
            #        f"raw contrib shape {raw_contribs.shape[0]} does not divide evenly by {M_max}.")
            # this is always true

            # we reshape this back into the batches:
            reshaped = tf.reshape(raw_contribs, (batch_size, M_max))
            # (batch_size, M_max)

            # apply the mask to zero out any artificial contributions from
            # the dummy padding features.

            pairwise_contribs = reshaped * tf.squeeze(masks, axis=2)

        # final reduction along axis 1 to get batch results:
        # shape (batch_size,)

        results = self.reduction_func(
            pairwise_contribs, **self.reduction_params)

        # calculate per-atom loss if specified and training:

        # this should correspond to the loss across the batch
        if (self.unitwise_loss_model is not None) and truth_supplied:
            self.add_loss(atomic_loss(y_true, results,
                                      sequence_lengths, self.unitwise_loss_model))

        return results

    def transform_dataset(self, X, y=None, sample_weight=None, batch_size=32, training=False):
        '''
        Makes the appropriate data transformations for the
        model's specified mode of operation.

        '''
        if isinstance(X, tf.data.Dataset):
            # currently not implemented
            raise NotImplementedError(
                "passing datasets directly not implemented, supply raw data instead.")
            # processed = X
            # N_batches = None

        else:
            # generates a Tensorflow.data.Dataset object from X, y.

            # how big each instance within a batch is.
            # needed for unstacking arrays in call()
            sequence_lengths = X.row_lengths(axis=1).numpy()

            # batch_sizes = chunksum(sequence_lengths, batch_size)
            num_instances = len(sequence_lengths)
            N_batches = -(num_instances // -batch_size)

            # print(sequence_lengths)
            # print(batch_sizes)
            # print(num_instances)
            # print(N_batches)

            if (y is not None) or training:
                # build a training (or validation) dataset with y included
                def __gen():
                    if sample_weight is not None:
                        for features, label, length, weights in zip(X, y, sequence_lengths, sample_weight):
                            yield (features, np.ones((features.shape[0], 1)), [length], label), label, weights
                    else:
                        for features, label, length in zip(X, y, sequence_lengths):
                            yield (features, np.ones((features.shape[0], 1)), [length], label), label
                if sample_weight is not None:
                    sample_weight_spec = (tf.TensorSpec(
                        shape=(1,), dtype=tf.float32),)
                    sample_weight_shape = (1,)
                else:
                    sample_weight_spec = ()
                    sample_weight_shape = ()

                signature = ((tf.TensorSpec(shape=(None, self.N_features), dtype=tf.float32),  # features
                              # masks (passed as a part of "Xtrain" to call())
                              tf.TensorSpec(shape=(None, 1),
                                            dtype=tf.float32),
                              # instance lengths (spans) for unstacking batched arrays
                              tf.TensorSpec(shape=(1,), dtype=tf.int32),
                              tf.TensorSpec(shape=(1,), dtype=tf.float32)),   # true labels for in-call loss determination
                             tf.TensorSpec(shape=(1,), dtype=tf.float32)) + sample_weight_spec  # labels and weights

                paddedshapes = (((None, self.N_features),  # features
                                 (None, 1),  # masks
                                 (1,),   # spans
                                 (1,)),  # true labels
                                (1,)) + sample_weight_shape  # labels and weights

            elif (X is not None) and y is None:
                # build prediction dataset with just x and aux. features.
                def __gen():
                    for features, length in zip(X, sequence_lengths):
                        yield (features, np.ones((features.shape[0], 1)), [length]),

                signature = ((tf.TensorSpec(shape=(None, self.N_features), dtype=tf.float32),  # features
                              # masks (passed as a part of "X" to call())
                              tf.TensorSpec(shape=(None, 1),
                                            dtype=tf.float32),
                              tf.TensorSpec(shape=(1,), dtype=tf.int32)),)  # instance lengths for unstacking batched arrays

                paddedshapes = (((None, self.N_features),  # features
                                 (None, 1),  # masks
                                 (1,)),)  # spans

            dataset = tf.data.Dataset.from_generator(
                generator=__gen,
                output_signature=signature)

        # handle ragged padding depending on initialization setting
        if self.ragged_processing:
            processed = dataset.ragged_batch(batch_size=batch_size)
        else:
            processed = dataset.padded_batch(
                batch_size=batch_size, padded_shapes=paddedshapes)  # labels

        return processed, N_batches

    def fit(self, x=None, y=None, **kwargs):
        '''
        overriden fit with fancy progress bar and pre-processing
        to handle ragged tensors nicely
        '''
        # intercept kwargs to construct a tf.dataset object if needed
        # do this for both training, validation sets (if present)

        batch_size = kwargs.pop("batch_size", 32)

        X_train = x
        y_train = y
        sample_weight_train = kwargs.pop("sample_weight", None)
        train_dataset, N_batches = self.transform_dataset(
            X_train, y_train, sample_weight_train, batch_size=batch_size)

        val_dataset = kwargs.pop("validation_data", None)
        if val_dataset:
            val_dataset, _ = self.transform_dataset(
                *val_dataset, batch_size=batch_size)

        progbar_callback = TqdmPaddedCallback(num_batches=N_batches)

        if "callbacks" not in kwargs or kwargs["callbacks"] is None:
            kwargs["callbacks"] = [progbar_callback]
        else:
            kwargs["callbacks"].append(progbar_callback)

        return super().fit(x=train_dataset, validation_data=val_dataset, **kwargs)

    def predict(self, x, batch_size=32, verbose='auto', steps=None, **kwargs):
        '''
        Model.predict, but overridden to do an initial data transform
        '''
        X, N_batches = self.transform_dataset(
            x, batch_size=batch_size, training=False)

        # progbar_callback = TqdmPaddedCallback(num_batches=N_batches, epochs=None)

        # if "callbacks" not in kwargs or kwargs["callbacks"] is None:
        #     kwargs["callbacks"] = [progbar_callback]
        # else:
        #     kwargs["callbacks"].append(progbar_callback)

        return super().predict(x=X, batch_size=batch_size, verbose=verbose, steps=steps, **kwargs)

    def evaluate(self, x=None, y=None, **kwargs):
        # to allow validation during training to still run
        if isinstance(x, tf.data.Dataset):
            return super().evaluate(x=x, **kwargs)

        batch_size = kwargs.pop("batch_size", 32)

        X_train = x
        y_train = y
        sample_weight_train = kwargs.pop("sample_weight", None)
        eval_dataset, N_batches = self.transform_dataset(
            X_train, y_train, sample_weight_train, batch_size=batch_size)

        return super().evaluate(x=eval_dataset, **kwargs)


class MLPNet(ReduceRegressor):
    def __init__(self, layers, N_features, ragged_processing=True, **kwargs):
        subnet = keras.Sequential(layers=[
            keras.Input(shape=(N_features,))]  # input layer takes in n_inputs number of symmetry function features
            + layers
            + [keras.layers.Dense(1, activation="linear")])
        super().__init__(subnet=subnet,
                         N_features=N_features,
                         reduction_func=tf.reduce_sum,
                         reduction_params={"axis": 1}, ragged_processing=ragged_processing, **kwargs)


class TqdmPaddedCallback(tqdm.keras.TqdmCallback):
    def __init__(self, epochs=None, num_batches=None, verbose=1,
                 tqdm_class=tqdm_auto, **tqdm_kwargs):
        super().__init__(epochs=epochs, verbose=1, tqdm_class=tqdm_class, **tqdm_kwargs)
        self.batches = num_batches
        if verbose == 1:
            self.batch_bar = tqdm_class(
                total=self.batches, unit='batch', leave=False)
            self.on_batch_end = self.bar2callback(
                self.batch_bar, pop=['batch', 'size'],
                delta=lambda logs: logs.get('size', 1))
