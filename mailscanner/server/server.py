'''
Server module, this can be used from the command line or via uWSGI
'''

import os

import connexion
import docopt

import mailscanner

PORT = os.environ.get('PORT', 5000)
DEBUG = os.environ.get('DEBUG', False)

SERVER_IN = os.path.dirname(os.path.abspath(__file__))
# WSGI module level variable
application = connexion.App(__name__, port=PORT, specification_dir=SERVER_IN)
application.add_api('api.yml')
mailscanner.server.replies.load_model_codec(
    os.path.join(SERVER_IN, '../../var/data/replies.weights'),
    os.path.join(SERVER_IN, '../../var/data/replies.pickle')
)

if __name__ == '__main__':
    application.run(debug=DEBUG)
