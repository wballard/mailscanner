'''
prepare-replies-dataset.py

Usage:
    prepare-replies-dataset.py <email_database> <dataset_text>

Prepare a text dataset from email replies, each line will be:
0 <tab> text of email without reply
1 <tab> text of email with reply
'''

import re

import docopt

import mailscanner


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    gdb = mailscanner.EmailDatabase(arguments['<email_database>'])
    replies = mailscanner.datasets.RepliedToDataset(gdb)
    scrub = re.compile('[\t\r\n]')
    with open(arguments['<dataset_text>'], mode='w', encoding='utf8') as dataset_text:
        for (reply, text) in replies.dataset:
            text = scrub.sub(' ', text)
            dataset_text.write('{0}\t{1}\n'.format(reply, text))
