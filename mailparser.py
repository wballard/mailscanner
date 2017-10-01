'''
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
'''
from bs4 import BeautifulSoup

import email
