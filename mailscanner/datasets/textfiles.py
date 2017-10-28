'''
Turn text files on disk into in memory tensors for use with machine learning.
'''

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

    At a given index `n` `labels[n]` is the corresponding label for `texts[n]`.

    >>> import mailscanner
    >>> dataset = mailscanner.datasets.LabeledTextFileDataset('./var/data/labeled.txt')
    >>> dataset.labels
    array([1, 0])
    >>> dataset.texts
    array([[112284,   8220,  64853, ...,      0,      0,      0],
           [107523,  82916, 185037, ...,      0,      0,      0]], dtype=int32)
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
       
        label_encoder = LabelEncoder()
        onehot_encoder = Pipeline([
            ('binarizer', LabelBinarizer()),
            ('onehot', OneHotEncoder())
        ])
        self.labels = label_encoder.fit_transform(label_buffer)
        # mildly tricky, need to wrap the array in an array
        self.one_hot_labels = onehot_encoder.fit_transform(self.labels).toarray()
        self.trigram = CharacterTrigramEmbedding()
        self.texts = self.trigram.sequencer.transform(text_buffer)
