'''
Adapters to store an individual user's email in a database.
'''

import os
import sqlite3

class EmailDatabase(sqlite3.Connection):
    '''
    Store email in raw RFC822 format with identifiers.
    '''

    def __init__(self, database_filename):
        '''
        Parameters
        ----------
        database_filename
            A string, where to store the file on disk. The folder must exist.
        '''
        if not os.path.exists(database_filename):
            creation_query = '''
            create table all_email(id text, body text);
            create unique index all_email_id on all_email(id);
            create table sent_email(id text, body text);
            create unique index sent_email_id on sent_email(id);
            '''
        else:
            creation_query = None
        super(EmailDatabase, self).__init__(database_filename)
        if creation_query:
            self.executescript(creation_query)
