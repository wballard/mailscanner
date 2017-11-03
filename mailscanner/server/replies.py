'''
Individual email handling methods.
'''

from ..datasets import LabeledTextFileDataset
from ..models import Ensemble

# preload this, it has a large tensor inside
# connexion only allows module level functions as handlers
# so module level caching of large data is required 
# -- beats loading in on every request!
MODEL = None
CODEC = None

def load_model_codec(path_to_weights, path_to_codec):
    '''
    Load up the module level variables for the codec/dataset
    and the trained machine learning model.
    '''
    global CODEC, MODEL
    print('loading codec from', path_to_codec)
    CODEC = LabeledTextFileDataset.load(path_to_codec)
    print('loading weights from', path_to_weights)
    MODEL = Ensemble(CODEC)
    MODEL.load_weights(path_to_weights)
    print(MODEL.summary())


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

    # text, sequenced as ngram, ready to be predicted
    body = body.decode('utf8')
    sequenced = CODEC.trigram.sequencer.transform([body])
    predicted = MODEL.predict(sequenced)
    decode = CODEC.decode_prediction(predicted[0])

    return {
        'label': decode[0],
        # cast off the numpy type
        'score': float(decode[1])
    }
