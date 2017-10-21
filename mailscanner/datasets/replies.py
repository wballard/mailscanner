'''
Create a dataset to learn which emails you are likely to reply to.
'''

from mailscanner import parse


class Replies:
    '''
    Build a dataset of emails that generated replies, along with a balanced number of
    negative samples that did not generate a reply.

    This visits every sent email in the provided database, extracts identifiers of emails that 
    generated replies, and hashes them.

    With a hash of all emails that generated replies in hand, all received emails are visited
    in order to extract the text of the email for each reply generating message. The very next
    non-reply-generating message encountered is used as a negative sample to offset and generate
    balanced classes.

    Attributes
    ----------
    dataset
        A list of (0|1, email text) tuples, where 1 indicates a reply, and 0 indicates no reply.
    '''

    def __init__(self, email_database):
        '''
        Parameters
        ----------
        email_database
            Visit this database to create training samples.
        '''
        replied_to = {}

        def is_a_reply(email):
            reply = parse(email).get('In-Reply-To', None)
            if reply:
                replied_to[reply] = True

        email_database.sent(is_a_reply)

        self.dataset = []

        def extract_replies(email):
            email = parse(email)
            if replied_to.get(email.get('Message-ID'), False):
                # a message that generated a reply!
                self.dataset.append((1, ' '.join(map(str, email.values()))))
                return
            # if we get here, this was not a reply, use it as a negative sample
            # if we have an odd number of entries to balance out
            if len(self.dataset) % 2 == 1:
                self.dataset.append((0, ' '.join(map(str, email.values()))))

        email_database.all(extract_replies)
