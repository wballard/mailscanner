'''
Ensemble text classifier.
'''

# data is loaded and preprocesed, now create a keras model to encode
import keras
from keras.layers import Dense
from vectoria import CharacterTrigramEmbedding

from ..layers import TimeDistributedSelfAttention, TimeStepReverse, SelfAttention

HIDDEN = 32
ACTIVATION = 'selu'
INITIALIZER = 'lecun_normal'


class Ensemble(keras.models.Model):
    '''
    This uses pretty much every available technique in parallel to classify text.

    >>> import mailscanner.models
    >>> m = mailscanner.models.Ensemble(32, 2)
    >>> m.save_weights('/tmp/m.model')
    >>> m = mailscanner.models.Ensemble(32, 2)
    >>> m = m.load_weights('/tmp/m.model')
    '''

    def __init__(self, maxlen=1024, classes=2):
        '''
        Parameters
        ----------
        sequence_length: int
            Number of input sequence steps.
        classes: int
            Number of total output classes, one hot.
        '''
        trigrams = CharacterTrigramEmbedding(maxlen=maxlen)

        inputs = keras.layers.Input(shape=(trigrams.maxlen,))

        # embedding to turn ngram identifiers dense
        embedded = trigrams.model(inputs)

        # plain old dense
        dense = Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(embedded)
        dense = keras.layers.Dropout(0.5)(dense)
        dense = Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(dense)
        dense = keras.layers.Dropout(0.5)(dense)

        # convolution to learn word and phrase like features
        conv = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(embedded)
        conv = keras.layers.MaxPooling1D(3)(conv)
        conv = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(conv)
        conv = keras.layers.MaxPooling1D(3)(conv)

        # recurrent with attention, this generates sequences
        # note this is GPU only, and keras
        # '>=2.0.9, it is shocking slow otherwise
        recurrent_forward = keras.layers.CuDNNLSTM(
            HIDDEN, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(conv)
        recurrent_backward = keras.layers.CuDNNLSTM(HIDDEN, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(TimeStepReverse()(conv))
        recurrent = keras.layers.Concatenate()(
            [recurrent_forward, recurrent_backward])

        # now attend to the most important
        self_attention = TimeDistributedSelfAttention(
            activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(conv)
        time_attention = SelfAttention(activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(conv)

        # now make a consistent shape and ensemble together as a stack, using global max pooling
        # to take out any remaining time steps and keep the strongest signals
        dense = keras.layers.GlobalMaxPooling1D()(dense)
        conv = keras.layers.GlobalMaxPooling1D()(conv)
        self_attention = keras.layers.GlobalMaxPooling1D()(self_attention)
        time_attention = keras.layers.GlobalMaxPooling1D()(time_attention)
        ensemble = keras.layers.Concatenate()(
            [dense, conv, recurrent, self_attention, time_attention])

        # dense before final output
        stack = Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(ensemble)
        stack = keras.layers.Dropout(0.5)(stack)
        stack = Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(
            0.), kernel_initializer=INITIALIZER)(stack)
        stack = keras.layers.Dropout(0.5)(stack)

        # softmax on two classes -- which map to our 0, 1 one hots
        outputs = Dense(classes, activation='softmax')(stack)

        super(Ensemble, self).__init__(inputs=inputs, outputs=outputs)
        self.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
