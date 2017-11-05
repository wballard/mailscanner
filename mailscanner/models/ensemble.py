'''
Ensemble text classifier.
'''

# data is loaded and preprocesed, now create a keras model to encode
import keras
from keras.layers import Dense
from vectoria import CharacterTrigramEmbedding

from ..layers import TimeDistributedSelfAttention, TimeStepReverse, SelfAttention

HIDDEN = 64
ACTIVATION = 'tanh'
INITIALIZER = 'he_normal'


class Ensemble(keras.models.Model):
    '''
    This uses pretty much every available technique in parallel to classify text.

    >>> import mailscanner
    >>> dataset = mailscanner.datasets.LabeledTextFileDataset('./var/data/labeled.txt') 
    >>> import mailscanner.models
    >>> m = mailscanner.models.Ensemble(dataset)
    >>> m.save_weights('/tmp/m.model')
    >>> m = mailscanner.models.Ensemble(dataset)
    >>> m = m.load_weights('/tmp/m.model')
    '''

    def __init__(self, source_dataset):
        '''
        Parameters
        ----------
        source_dataset: `LabeledTextFileDataset`
            Contains the source and target data to derive the shape and encoding of the model.
        '''
        trigrams = source_dataset.trigram

        inputs = keras.layers.Input(shape=(trigrams.maxlen,))

        # embedding to turn ngram identifiers dense
        embedded = trigrams.build_model()(inputs)

        # plain old dense
        dense = Dense(HIDDEN, 
            activation=ACTIVATION, 
            kernel_regularizer=keras.regularizers.l2(0.), 
            kernel_initializer=INITIALIZER)(embedded)
        dense = keras.layers.Dropout(0.5)(dense)
        dense = Dense(HIDDEN, 
            activation=ACTIVATION, 
            kernel_regularizer=keras.regularizers.l2(0.), 
            kernel_initializer=INITIALIZER)(dense)
        dense = keras.layers.Dropout(0.5)(dense)

        # convolution to learn word and phrase like features
        conv = keras.layers.Conv1D(HIDDEN,
            3,
            activation=ACTIVATION,
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(embedded)
        conv = keras.layers.Conv1D(HIDDEN,
            3,
            activation=ACTIVATION,
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(conv)
        conv = keras.layers.MaxPooling1D(3)(conv)
        conv = keras.layers.Conv1D(HIDDEN,
            3,
            activation=ACTIVATION,
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(conv)
        conv = keras.layers.Conv1D(HIDDEN,
            3,
            activation=ACTIVATION,
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(conv)
        conv = keras.layers.MaxPooling1D(3)(conv)

        # recurrent with attention, this generates sequences
        recurrent_forward = keras.layers.LSTM(HIDDEN, 
            kernel_regularizer=keras.regularizers.l2(0.), 
            kernel_initializer=INITIALIZER)(conv)
        recurrent_backward = keras.layers.LSTM(HIDDEN, 
            kernel_regularizer=keras.regularizers.l2(0.), 
            kernel_initializer=INITIALIZER)(TimeStepReverse()(conv))
        recurrent = keras.layers.Concatenate()(
            [recurrent_forward, recurrent_backward])

        # now attend to the most important
        time_attention = TimeDistributedSelfAttention(activation=ACTIVATION,
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(conv)
        self_attention = SelfAttention(activation=ACTIVATION,
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(conv)

        # now make a consistent shape and ensemble together as a stack, using global max pooling
        # to take out any remaining time steps and keep the strongest signals
        dense = keras.layers.GlobalMaxPooling1D()(dense)
        conv = keras.layers.GlobalMaxPooling1D()(conv)
        self_attention = keras.layers.GlobalMaxPooling1D()(self_attention)
        time_attention = keras.layers.GlobalMaxPooling1D()(time_attention)
        ensemble = keras.layers.Concatenate()(
            [dense, conv, recurrent, self_attention, time_attention])

        # dense before final output
        stack = Dense(HIDDEN, 
            activation=ACTIVATION, 
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(ensemble)
        stack = keras.layers.Dropout(0.5)(stack)
        stack = Dense(HIDDEN, 
            activation=ACTIVATION, 
            kernel_regularizer=keras.regularizers.l2(0.),
            kernel_initializer=INITIALIZER)(stack)
        stack = keras.layers.Dropout(0.5)(stack)

        # softmax on the numbered of labaled classes -- which map to our 0, 1 one hots
        outputs = Dense(len(source_dataset.label_encoder.classes_), activation='softmax')(stack)

        super(Ensemble, self).__init__(inputs=inputs, outputs=outputs)
        self.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )