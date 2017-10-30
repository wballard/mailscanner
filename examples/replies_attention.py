import mailscanner
replies = mailscanner.datasets.LabeledTextFileDataset('./replies.txt')

# one hot encode the labels
targets = replies.one_hot_labels

# now set up sequencing and embedding
sources = replies.texts

HIDDEN = 128
ACTIVATION = 'relu'


# data is loaded and preprocesed, now create a keras model to encode
import keras

inputs = keras.layers.Input(shape=(replies.trigram.maxlen,))
# embedding to turn ngram identifiers dense
embedded = replies.trigram.model(inputs)
# convolution to learn word and phrase like features
stack = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION)(embedded)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION)(stack)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Conv1D(HIDDEN, 3, activation=ACTIVATION)(stack)
stack = keras.layers.MaxPooling1D(3)(stack)
# recurrent layer -- read the word like structures in time series order
# note this is GPU only, and keras
# '>=2.0.9, it is shocking slow otherwise
recurrent_forward = keras.layers.CuDNNLSTM(HIDDEN, return_sequences=True)(stack)
recurrent_backward = keras.layers.CuDNNLSTM(HIDDEN, return_sequences=True)(mailscanner.layers.TimeStepReverse()(stack))
recurrent = keras.layers.Concatenate()([recurrent_forward, recurrent_backward])
# now attend to the most important
attention = mailscanner.layers.TimeDistributedSelfAttention(activation=ACTIVATION)(recurrent)
# dense before final output
stack = keras.layers.Dense(HIDDEN, activation=ACTIVATION)(attention)
stack = keras.layers.Dropout(0.5)(stack)
stack = keras.layers.Dense(HIDDEN, activation=ACTIVATION)(stack)
stack = keras.layers.Dropout(0.5)(stack)
# softmax on two classes -- which map to our 0, 1 one hots
stack = keras.layers.Flatten()(stack)
outputs = keras.layers.Dense(2, activation='softmax')(stack)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()
model.fit(
    x=sources,
    y=targets,
    validation_split=0.05,
    batch_size=32,
    epochs=128
)

