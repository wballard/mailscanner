'''
Individual email handling methods.
'''

from ..parser import parse
from ..datasets import LabeledTextFileDataset
from ..models import Ensemble
from vectoria import CharacterTrigramEmbedding

# preload this, it has a large tensor inside
MODEL = None
CODEC = None


def load_model(path_to_weights):
    '''
    Restore weights into the model.
    '''


def load_model_codec(path_to_weights, path_to_codec):
    global CODEC, MODEL
    print('loading codec from', path_to_codec)
    CODEC = LabeledTextFileDataset.load(path_to_codec)
    MODEL = Ensemble(CODEC)
    print('loading weights from', path_to_weights)
    MODEL.load_weights(path_to_weights)
    print(MODEL.summary())
    print(CODEC.trigram)


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
    
