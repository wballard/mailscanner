'''
Mailscanner main, import and expose classes here.
'''

from pkg_resources import get_distribution

from . import datasets
from . import layers
from .databases import EmailDatabase
from .parser import parse
from .sources import GmailSource

__version__ = get_distribution('mailscanner').version
