import mailscanner
replies = mailscanner.datasets.LabeledTextFileDataset('./replies.txt')

# one hot encode the labels
targets = replies.one_hot_labels

# now set up sequencing and embedding
sources = replies.texts

HIDDEN = 32
ACTIVATION = 'selu'
INITIALIZER = 'lecun_normal'

# data is loaded and preprocesed, now create a keras model to encode
import keras

inputs = keras.layers.Input(shape=(replies.trigram.maxlen,))

# embedding to turn ngram identifiers dense
embedded = replies.trigram.model(inputs)


# plain old dense
dense = keras.layers.Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(embedded)
dense = keras.layers.Dropout(0.5)(dense)
dense = keras.layers.Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(dense)
dense = keras.layers.Dropout(0.5)(dense)

# convolution to learn word and phrase like features
conv = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(embedded)
conv = keras.layers.MaxPooling1D(3)(conv)
conv = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(conv)
conv = keras.layers.MaxPooling1D(3)(conv)

# recurrent with attention, this generates sequences
# note this is GPU only, and keras
# '>=2.0.9, it is shocking slow otherwise
recurrent_forward = keras.layers.CuDNNLSTM(HIDDEN, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(conv)
recurrent_backward = keras.layers.CuDNNLSTM(HIDDEN, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(mailscanner.layers.TimeStepReverse()(conv))
recurrent = keras.layers.Concatenate()([recurrent_forward, recurrent_backward])

# now attend to the most important
self_attention = mailscanner.layers.TimeDistributedSelfAttention(activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(conv)
time_attention = mailscanner.layers.SelfAttention(activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(conv)

# now make a consistent shape and ensemble together as a stack, using global max pooling
# to take out any remaining time steps and keep the strongest signals
dense = keras.layers.GlobalMaxPooling1D()(dense)
conv = keras.layers.GlobalMaxPooling1D()(conv)
self_attention = keras.layers.GlobalMaxPooling1D()(self_attention)
time_attention = keras.layers.GlobalMaxPooling1D()(time_attention)
ensemble = keras.layers.Concatenate()([dense, conv, recurrent, self_attention, time_attention])


# dense before final output
stack = keras.layers.Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(ensemble)
stack = keras.layers.Dropout(0.5)(stack)
stack = keras.layers.Dense(HIDDEN, activation=ACTIVATION, kernel_regularizer=keras.regularizers.l2(0.), kernel_initializer=INITIALIZER)(stack)
stack = keras.layers.Dropout(0.5)(stack)

# softmax on two classes -- which map to our 0, 1 one hots
outputs = keras.layers.Dense(2, activation='softmax')(stack)

model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# model was impressive in overfitting -- hitting 100% accuracy in train
# so -- change the mix shift of the data, more for testing, less data for overfitting
# moving even less data to validation, meaning more training data --> less overfitting
model.fit(
    x=sources,
    y=targets,
    validation_split=0.01,
    batch_size=128,
    epochs=256
)

