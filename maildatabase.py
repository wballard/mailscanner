'''
Classes to connect and download email to a database for further processing.
'''

import imaplib
import os
import sqlite3

from tqdm import tqdm

# you may have a LOT of email, so take this limit up
imaplib._MAXLINE = 16 * 1024 * 1024


class MailSource:
    '''
    Base mail connector.

    Errors will propagate from imaplib, raising imaplib.IMAP4.error.
    '''

    def __init__(self, host, username, password):
        self.mail = imaplib.IMAP4_SSL(host)
        self.mail.login(username, password)

    def identifiers(self, folder):
        '''
        General email identity fetcher.

        Parameters
        ----------
        folder
            A string naming the folder to fetch all identifiers.

        Returns
        -------
        list
            A list of email identifiers.

        '''
        self.mail.select(folder)
        _, data = self.mail.uid('search', None, "ALL")
        email_identifiers = data[0].split()
        return email_identifiers

    def __getitem__(self, email_identifier):
        '''
        Fetch a single email by identifier.

        Parameters
        ----------
        email_identifier
            A single email identification string.
        '''
        _, data = self.mail.uid('fetch', email_identifier, '(RFC822)')
        return data[0][1]


class GmailSource(MailSource):
    '''
    Connect to Gmail for Email.

    This connection will use single factor authentication, so you will need to
    'Allow less secure apps' on https://myaccount.google.com/security.
    '''

    def __init__(self, username, password):
        '''
        Connect to gmail via IMAP.

        Parameters
        ----------
        username
            Your gmail address.
        password
            Your gmail password.
        '''
        super(GmailSource, self).__init__('imap.gmail.com', username, password)

    def all(self):
        '''
        All inbound email.
        '''
        return self.identifiers('"[Gmail]/All Mail"')

    def sent(self):
        '''
        Outbound mail you have sent.
        '''
        return self.identifiers('"[Gmail]/Sent Mail"')


class EmailDatabase:
    '''
    Store email in raw RFC822 format with identifiers.
    '''

    def __init__(self, database_filename, email_source):
        '''
        Parameters
        ----------
        database_filename
            A string, where to store the file on disk. The folder must exist.
        email_source
            A subclass of MailSource with an all() and sent() method.
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
        self.email_database = sqlite3.connect(database_filename)
        if creation_query:
            self.email_database.executescript(creation_query)
        self.email_source = email_source

    def download(self):
        '''
        Download all email. This uses a two pass algorithm to allow restart and
        catch up to avoid the pain of downloading every email every time.

        Pass 1 - get all identifiers of all email, storing them in the database.
        Pass 2 - for all identifiers without a body, download and store the body in the database.
        '''

        sources = [('all_email', self.email_source.all),
                   ('sent_email', self.email_source.sent)]
        for table, source in sources:
            # pass 1 -- identifiers
            cursor = self.email_database.cursor()
            identity_save = '''
                insert or ignore into {0} (id, body)
                values (?, null)
            '''.format(table)

            for identifier in tqdm(source(), desc=table, unit='id'):
                cursor.execute(identity_save, (identifier.decode('utf8'),))
            self.email_database.commit()

            # pass 2 -- fill in email
            cursor = self.email_database.cursor()
            identity_read = '''
                select id from {0}
                where body is null
            '''.format(table)
            email_save = '''
                update {0}
                set body = ?
                where id = ?
            '''.format(table)
            cursor.execute(identity_read)
            for row in tqdm(cursor.fetchall(), desc=table, unit='email'):
                identifier = row[0]
                body = self.email_source[identifier]
                try:
                    cursor.execute(email_save, (body.decode('utf8'), identifier))
                except UnicodeDecodeError:
                    cursor.execute(email_save, ('', identifier))
                self.email_database.commit()



if __name__ == '__main__':
    g = GmailSource(os.environ['GMAIL_ADDRESS'], os.environ['GMAIL_PASSWORD'])
    gdb = EmailDatabase('gmail.db', g)
    gdb.download()
