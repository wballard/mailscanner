'''
Layers that implement attention mechanisms.
'''
import keras
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec, Layer
import keras.backend as K


class SelfAttention(Layer):
    """
    Implements an self attention mechanism over time series data, weighting the
    input time series by a learned, softmax scaled attention matrix.

    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
    # Input shape
        3D tensor with shape: `(batch_size, time_step, dimensions)
    # Output shape
        3D tensor with shape: `(batch_size, time_step, scaled_dimensions)`.
    """

    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SelfAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        time_steps = input_shape[1]
        dimensions = input_shape[2]

        self.attention = keras.models.Sequential(name='attention')
        # starting off, each element of the batch is (time_steps, dimensions)
        # turn this into (time_steps, 1)

        # attention matrix, this is the main thing being learned
        self.attention.add(keras.layers.Dense(dimensions,
                                              input_shape=(
                                                  time_steps, dimensions,),
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint))
        self.attention.add(keras.layers.Activation(self.activation))
        # now convert to an attention vector
        self.attention.add(keras.layers.Dense(1,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint))
        # make an attention vector
        self.attention.add(keras.layers.Flatten())
        self.attention.add(keras.layers.Activation('softmax'))
        # repeat this time step weighting for each dimensions
        self.attention.add(keras.layers.RepeatVector(dimensions))
        # reshape to be (time_steps, dimensions)
        self.attention.add(keras.layers.Permute([2, 1]))

        # not using add_weight, so update the weighs manually
        self.trainable_weights = self.attention.trainable_weights
        self.non_trainable_weights = self.attention.non_trainable_weights

        # all done
        self.built = True

    def call(self, inputs):
        # build the attention matrix
        attention = self.attention(inputs)
        # apply the attention matrix with element wise multiplication
        return keras.layers.Multiply()([inputs, attention])

    def compute_output_shape(self, input_shape):
        # there is no change in shape, the values are just weighted
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        return dict(config)


class TimeDistributedSelfAttention(Layer):
    """
    Implements an self attention mechanism over time series data, weighting the
    input time series by a learned, softmax scaled attention matrix.

    This will lean attention as a time distributed repetition of a dense
    neural network over the time steps of the input.

    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
    # Input shape
        3D tensor with shape: `(batch_size, time_step, dimensions)
    # Output shape
        3D tensor with shape: `(batch_size, time_step, scaled_dimensions)`.
    """

    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TimeDistributedSelfAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        dimensions = input_shape[2]
        # build a keras model that will be used inside this layer
        # defining a layer that is itself a model!
        timed = keras.models.Sequential(name='per_time_step')
        timed.add(keras.layers.Dense(dimensions,
                                     input_shape=(dimensions,),
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     kernel_constraint=self.kernel_constraint))
        timed.add(keras.layers.Activation(self.activation))
        timed.add(keras.layers.Dense(dimensions,
                                     input_shape=(dimensions,),
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     kernel_constraint=self.kernel_constraint))
        timed.add(keras.layers.Activation(self.activation))
        self.timed = keras.layers.TimeDistributed(timed)

        # not using add_weight, so update the weighs manually
        self.trainable_weights = self.timed.trainable_weights
        self.non_trainable_weights = self.timed.non_trainable_weights
        # all done
        self.built = True

    def call(self, inputs):
        # run the time distributed model over the input
        encoded = self.timed(inputs)
        # now take the product of the encoding and the original input, combining
        self_attended = K.batch_dot(
            inputs, K.permute_dimensions(encoded, (0, 2, 1)))
        # 2D softmax, this ends up being a multiple dimension attention
        # with weights in each time step normalizing to probabilities
        attention = K.softmax(self_attended)
        # make sure the softmax is over the time series
        attention = K.permute_dimensions(attention, (0, 2, 1))
        # and finally, weight the input with the attention
        return K.batch_dot(attention, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        return dict(config)
