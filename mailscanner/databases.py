'''
Adapters to store an individual user's email in a database.
'''

import os
import sqlite3

from tqdm import tqdm


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


    def sent(self, visitor, verbose=True):
        '''
        Visit all sent emails.

        Parameters
        ----------
        visitor
            A callable that receives each email text.
        '''
        cursor = self.cursor()
        cursor.execute('select count(*) from sent_email')
        count = cursor.fetchall()[0][0]
        cursor.execute('select body from sent_email')
        for row in tqdm(cursor.fetchall(), total=count, desc="Sent", unit='email', disable=(not verbose)):
            visitor(row[0])

    def all(self, visitor, verbose=True):
        '''
        Visit all emails.

        Parameters
        ----------
        visitor
            A callable that receives each email text.
        '''
        cursor = self.cursor()
        cursor.execute('select count(*) from all_email')
        count = cursor.fetchall()[0][0]
        cursor.execute('select body from all_email')
        for row in tqdm(cursor, total=count, desc="All", unit='email', disable=(not verbose)):
            visitor(row[0])