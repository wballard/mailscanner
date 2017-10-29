'''
Layers for the reversing of input tensors.
'''

import keras

class TimeStepReverse(keras.layers.Layer):
    """
    A custom keras layer to reverse a tensor along the first
    non batch dimension, assumed to be the time step.
    
    # Input shape
        nD tensor with shape: `(batch_size, time_step, ...)`.
    # Output shape
        nD tensor with shape: `(batch_size, time_step, ...)`.
    """

    def call(self, tensor):
        """
        Use the backed to reverse.
        """
        return keras.backend.reverse(tensor, 1)

    def compute_output_shape(self, input_shape):
        """
        No change in shape.
        """
        return input_shape