'''
prepare-replies-model.py

Usage:
    prepare-replies-model.py <replies_text_dataset> <output_model_weights> <output_model_codec>

Take a replies labled text dataset and then train and save a useable
model.
'''

import pickle

import docopt
import keras

import mailscanner

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    replies = mailscanner.datasets.LabeledTextFileDataset(
        arguments['<replies_text_dataset>'])
    targets = replies.one_hot_labels
    sources = replies.texts
    model = mailscanner.models.Ensemble(replies)
    
    print(model.summary())
    #saved encoder
    replies.save(arguments['<output_model_codec>'])

    # run with callback to save the best performing weights
    save_best_weights = keras.callbacks.ModelCheckpoint(
        arguments['<output_model_weights>'], 
        monitor='val_acc',
        save_best_only=True, 
        save_weights_only=True, 
        verbose=True)
    early_exit = keras.callbacks.EarlyStopping(monitor='val_acc', patience=4)
    hist = model.fit(
        x=sources,
        y=targets,
        validation_split=0.1,
        batch_size=128,
        epochs=256,
        callbacks=[save_best_weights, early_exit]
    )
    print(hist.history['val_acc'])
