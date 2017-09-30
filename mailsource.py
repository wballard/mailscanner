'''
Classes to connect via email to various email
providers.

The returned emails are parsed into data and metadata/
'''

import os
import imaplib
import email
from bs4 import BeautifulSoup


class MailSource:
    '''
    Base mail connector. 
    '''

    def __init__(self, host, username, password):
        self.mail = imaplib.IMAP4_SSL(host)
        self.mail.login(username, password)

    def all_mail(self):
        '''
        Iterate and yield all email.
        '''
        self.mail.select("inbox")
        result, data = self.mail.uid('search', None, "ALL") # search and return uids instead
        print(result, len(data[0].split()))
        latest_email_uid = data[0].split()[-1]
        result, data = self.mail.uid('fetch', latest_email_uid, '(RFC822)')

        # get this into a dict type format for easy access 
        message = email.message_from_string(data[0][1].decode('utf8'))
        message_text = []
        for part in message.walk():
            content_type = part.get_content_type() 
            print(content_type)
            if content_type == 'text/plain':
                message_text.append(part.get_payload())
            if content_type == 'text/html':
                soup = BeautifulSoup(part.get_payload(), 'html.parser')
                message_text.append(soup.get_text())
        final_email = dict(message.items())
        final_email['text'] = '\n'.join(message_text)
        print(result, final_email)


class GmailSource(MailSource):
    '''
    Connect to Gmail for Email.

    This connection will use single factor authentication, so you will need to 
    'Allow less secure apps' on https://myaccount.google.com/security.
    '''

    def __init__(self, username, password):
        super(GmailSource, self).__init__('imap.gmail.com', username, password)


if __name__ == '__main__':
    g = GmailSource(os.environ['GMAIL_ADDRESS'], os.environ['GMAIL_PASSWORD'])
    g.all_mail()
