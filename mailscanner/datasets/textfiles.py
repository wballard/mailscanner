'''
Turn text files on disk into in memory tensors for use with machine learning.
'''

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from smart_open import smart_open

from vectoria import CharacterTrigramEmbedding


class LabeledTextFileDataset:
    '''
    Read text files with one string per line, of the form:
    <label> <tab> <text>

    Attributes
    ----------
    labels
        A binay encoded array of the labels
    one_hot_labels
        One hot encoding of labels
    texts
        A 2-d tensor of string and ngram positional sequence encodings.
    trigram
        A `CharacterTrigramEmbedding` instance, where you can get the embedding model.

    At a given index `n` `labels[n]` is the corresponding label for `texts[n]`.

    >>> import mailscanner
    >>> dataset = mailscanner.datasets.LabeledTextFileDataset('./var/data/labeled.txt')
    >>> dataset.labels
    array([1, 0])
    >>> dataset.texts
    array([[112284,   8220,  64853, ...,      0,      0,      0],
           [107523,  82916, 185037, ...,      0,      0,      0]], dtype=int32)
    >>> dataset.decode_prediction([0.25, 0.75])
    ('Good', 0.75)
    >>> dataset.save('/tmp/labeled.pickle')
    >>> readback = mailscanner.datasets.LabeledTextFileDataset.load('/tmp/labeled.pickle')
    '''

    def __init__(self, textfile_path):
        '''
        Read the text, and separate it.

        Parameters
        ----------
        textfile_path
            A string path, which is passed to `smart_open`, so this can can a local file
            or even an S3 url.
        '''
        label_buffer = []
        text_buffer = []
        for line in smart_open(textfile_path):
            label, text = line.decode('utf8').split('\t')
            label_buffer.append(label)
            text_buffer.append(text.strip())

        self.label_encoder = label_encoder = LabelEncoder()
        self.label_binarizer = label_binarizer = LabelBinarizer()
        self.onehot_encoder = onehot_encoder = OneHotEncoder()
        self.labels = label_encoder.fit_transform(label_buffer)
        self.one_hot_labels = OneHotEncoder().fit_transform(LabelBinarizer().fit_transform(self.labels)).toarray()
        # mildly tricky, need to wrap the array in an array
        strings = StringsDataset(text_buffer)
        self.trigram = strings.trigram
        self.texts = strings.texts

    def decode_prediction(self, one_hot):
        '''
        Given a set of one-hot encoded values, return the labels and prediction values.
        '''
        winner = np.argmax(one_hot)
        label = self.label_encoder.inverse_transform([winner])
        return (label[0], one_hot[winner])

    def save(self, path_to_file):
        '''
        Save this dataset off to a pickle.
        '''
        pickle.dump(self, open(path_to_file, 'wb'))

    @classmethod
    def load(cls, path_to_file):
        '''
        Load up a pickled dataset.
        '''
        return pickle.load(open(path_to_file, 'rb'))


class StringsDataset:
    '''
    Turn a list of strings into a 2d tensor of ngram sequence identifiers.

    Attributes
    ----------
    texts
        A 2-d tensor of string and ngram positional sequence encodings.
    trigram
        A `CharacterTrigramEmbedding` instance, where you can get the embedding model.
    '''

    def __init__(self, strings):
        '''
        Parameters
        ----------
        strings
            A list of strings to transform.
        '''
        self.trigram = CharacterTrigramEmbedding()
        self.texts = self.trigram.sequencer.transform(strings)
