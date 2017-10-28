import mailscanner
replies = mailscanner.datasets.LabeledTextFileDataset('./replies.txt')

# one hot encode the labels
targets = replies.one_hot_labels

# now set up sequencing and embedding
sources = replies.texts


# data is loaded and preprocesed, now create a keras model to encode
# Here is a very simple -- dense -- model,
# not taking into any real consideration the structure of text,
# or the sequential nature of words or ngrams.
import keras
inputs = keras.layers.Input(shape=(replies.trigram.maxlen,))
embedded = replies.trigram.model(inputs)
dense_one = keras.layers.Dense(128, activation='relu')(embedded)
dense_two = keras.layers.Dense(128, activation='relu')(dense_one)
# softmax on two classes -- which map to our 0, 1 one hots
flattened = keras.layers.Flatten()(dense_two)
outputs = keras.layers.Dense(2, activation='softmax')(flattened)
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
    epochs=16
)
