'''
Classes to connect and download email to a database for further processing.
'''

import imaplib
import os

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
   
    def download(self, email_database):
        '''
        Download all email. This uses a two pass algorithm to allow restart and
        catch up to avoid the pain of downloading every email every time.

        Pass 1 - get all identifiers of all email, storing them in the database.
        Pass 2 - for all identifiers without a body, download and store the body in the database.

        Parameters
        ----------
        email_database
            An `EmailDatabase` instance. Email content will be stored here.
        '''

        sources = [('all_email', self.all),
                   ('sent_email', self.sent)]
        for table, source in sources:
            # pass 1 -- identifiers
            cursor = email_database.cursor()
            identity_save = '''
                insert or ignore into {0} (id, body)
                values (?, null)
            '''.format(table)

            for identifier in tqdm(source(), desc=table, unit='id'):
                cursor.execute(identity_save, (identifier.decode('utf8'),))
            email_database.commit()

            # pass 2 -- fill in email
            cursor = email_database.cursor()
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
                body = self[identifier]
                try:
                    cursor.execute(email_save, (body.decode('utf8'), identifier))
                except UnicodeDecodeError:
                    cursor.execute(email_save, ('', identifier))
                email_database.commit()


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
