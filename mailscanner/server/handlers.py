'''
Individual email handling methods.
'''

from ..parser import parse
from ..datasets import StringsDataset
from vectoria import CharacterTrigramEmbedding

# preload this, it has a large tensor inside
TRIGRAMS = CharacterTrigramEmbedding()

def rfc822(body):
    '''
    Parameters
    ----------
    body
        An entire RFC822 string.

    Returns
    -------
    string
        JSON string encoding the classification result.
    '''

    #text, sequenced as ngram, ready to be predicted
    body = body.decode('utf8')
    sequenced = TRIGRAMS.sequencer.transform([body])
    print(sequenced)

    return {
        'label': None
    }