'''
Turn text files on disk into in memory tensors for use with machine learning.
'''

from smart_open import smart_open

class LabeledTextFileDataset:
    '''
    Read text files with one string per line, of the form:
    <label> <tab> <text>

    Attributes
    ----------
    labels
        A list of label values
    texts
        A list of text values

    At a given index `n` `labels[n]` is the corresponding label for `texts[n]`.
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
        self.labels = []
        self.texts = []
        for line in smart_open(textfile_path):
            label, text = line.decode('utf8').split('\t')
            self.labels.append(label)
            self.texts.append(text)
