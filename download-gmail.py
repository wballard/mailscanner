'''
download-gmail.py

Usage:
    download-gmail.py <database>

You will be prompted for a gmail username and password.
'''

import getpass
import os

import docopt

import mailscanner

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    username = input("GMail Address:")
    password = getpass.getpass()
    g = mailscanner.GmailSource(username, password)
    gdb = mailscanner.EmailDatabase(arguments['<database>'])
    g.download(gdb)
