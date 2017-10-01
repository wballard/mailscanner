'''
Given RFC822 text, parse email into a more structured format.
'''

import email
import pprint
from pathlib import Path


def parse(rfc822_string):
    '''
    Parameters
    ----------
    rfc822_string
        Text of an email in RFC822 format.

    Returns
    -------
    list
        A list of tuples, in (name, value) format for each header,
        along with a (text, ""), and (html, "") tuple with the body content.
    '''
    message = email.message_from_string(rfc822_string)
    headers = list(map(parse_headers, message.items()))
    message_text = []
    message_html = []
    for part in message.walk():
        content_type = part.get_content_type()
        if content_type == 'text/plain':
            message_text.append(part.get_payload())
        if content_type == 'text/html':
            message_html.append(part.get_payload())
    text = ('text', '\n'.join(message_text))
    html = ('html', '\n'.join(message_html))
    return headers + [text] + [html]


def parse_headers(header_tuple):
    '''
    Given a header (name, value) tuple, do additional parsing
    for more structure on email addresses and dates.

    Parameters
    ----------
    header_tuple
        A (name, value) pair of strings.

    Returns
    -------
    tuple
        A (name, value) tuple, where value will be parsed to
        a more structured tuple than merely a string.
    '''
    name, value = header_tuple
    if name in ['To', 'From', 'Delivered-To']:
        value = email.utils.parseaddr(value)
    if name == 'Date':
        value = email.utils.parsedate_tz(value)
    return (name, value)


if __name__ == '__main__':
    parsed = parse(Path('./var/data/email.txt').read_text())
    pprint.pprint(parsed)
