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
                 kernel_initializer='he_normal',
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

        # learn the most important time steps here by factoring out dimensions
        self.alignment_kernel = self.add_weight(shape=(dimensions, time_steps),
                                                initializer=self.kernel_initializer,
                                                name='alignment_kernel',
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)

        # learn multiple attentions here
        self.attention_kernel = self.add_weight(shape=(time_steps, 1),
                                                initializer=self.kernel_initializer,
                                                name='attention_kernel',
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        # all done
        self.built = True

    def call(self, inputs):
        # transpose to get the time dimensions facing each other and will result
        # in a square time step matrix
        alignment = K.dot(inputs, self.alignment_kernel)
        # with an optional activation function here to introduce a non-linearity
        if self.activation is not None:
            alignment = self.activation(alignment)
        # get to a 'vector' of (batch_size, time_steps, 1), really squeeze it to a vector of (batch_size, time_steps)
        attention = K.squeeze(K.dot(alignment, self.attention_kernel), 2)
        # normalize the attention matrix with softmax
        attention = K.softmax(attention)
        # back up to (batch, time_steps, 1)
        attention = K.expand_dims(attention)
        # now we take element wise multiplication to weight the inputs
        # by the learned attention
        output = inputs * attention
        return output

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
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultipleSelfAttention(Layer):
    """
    Implements an self attention mechanism over time series data, weighting the
    input time series by a learned, softmax scaled attention matrix.

    This will lean multiple attention heads, one for each time step.

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
        super(MultipleSelfAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

    def build(self, input_shape):
        time_steps = input_shape[1]
        dimensions = input_shape[2]

        # learn the most important time steps here
        self.alignment_kernel = self.add_weight(shape=(dimensions, time_steps),
                                                initializer=self.kernel_initializer,
                                                name='alignment_kernel',
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)

        # learn multiple attentions here
        self.multiple_attention_kernel = self.add_weight(shape=(time_steps, time_steps),
                                                         initializer=self.kernel_initializer,
                                                         name='multiple_attention_kernel',
                                                         regularizer=self.kernel_regularizer,
                                                         constraint=self.kernel_constraint)
        # all done
        self.built = True

    def call(self, inputs):
        # transpose to get the time dimensions facing each other and will result
        # in a square time step matrix
        alignment = K.dot(inputs, self.alignment_kernel)
        # with an optional activation function here to introduce a nonlinerity
        if self.activation is not None:
            alignment = self.activation(alignment)
        # apply multiple attention here, this again results in a square time step matrix
        multiple = K.dot(alignment, self.multiple_attention_kernel)
        # normalize the attention matrix with a 2 dimensional softmax
        attention = K.softmax(multiple)
        # now we have to batch shaped objects, so batch dot is required
        # to maintain the batch dimension
        output = K.batch_dot(attention, inputs)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
