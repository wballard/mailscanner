import mailscanner
replies = mailscanner.datasets.LabeledTextFileDataset('./replies.txt')

# one hot encode the labels
targets = replies.one_hot_labels

# now set up sequencing and embedding
sources = replies.texts

HIDDEN = 128

# data is loaded and preprocesed, now create a keras model to encode
# Here is a very simple -- dense -- model,
# not taking into any real consideration the structure of text,
# or the sequential nature of words or ngrams.
import keras

class Reverse(keras.layers.Layer):
    """
    A custom keras layer to reverse a tensor.
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

inputs = keras.layers.Input(shape=(replies.trigram.maxlen,))
# embedding to turn ngram identifiers dense
embedded = replies.trigram.model(inputs)
stack = keras.layers.Conv1D(HIDDEN, 3, activation='relu')(embedded)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Conv1D(HIDDEN, 3, activation='relu')(stack)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Conv1D(HIDDEN, 3, activation='relu')(stack)
stack = keras.layers.MaxPooling1D(3)(stack)
stack = keras.layers.Dropout(0.5)(stack)
# recurrent layer -- read the word like structures in time series order
# note this is GPU only, and keras
# '>=2.0.9, it is shocking slow otherwise
recurrent_forward = keras.layers.CuDNNLSTM(HIDDEN)(stack)
recurrent_backward = keras.layers.CuDNNLSTM(HIDDEN)(Reverse()(stack))
recurrent = keras.layers.Concatenate()([recurrent_forward, recurrent_backward])
recurrent = keras.layers.Dropout(0.5)(recurrent)
# dense before final output
stack = keras.layers.Dense(HIDDEN, activation='relu')(recurrent)
stack = keras.layers.Dropout(0.5)(stack)
stack = keras.layers.Dense(HIDDEN, activation='relu')(stack)
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
model.fit(
    x=sources,
    y=targets,
    validation_split=0.05,
    batch_size=32,
    epochs=32
)
