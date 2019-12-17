'''   _____________    ______________
-----/ ____/ __  / /_ / / ____/_  __/
----/ /   / / / / /|// / __/   / /
---/ /___/ /_/ / /  / / /___  / /
--/_____/____ /_/  /_/_____/ /_/

Created by interns in the REHS program under the
mentorship of Dr. Martin Kandes:
 - Nicholas Clark
 - Roxane Martin
 - Dustin Wu

Run pip install -r requirements.txt to install all neccessary libraries.
See bin/help.txt for more info.

sources: 
http://naelshiab.com/tutorial-send-email-python/
https://stackoverflow.com/questions/31433633/reply-to-email-using-python-3-4
'''

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.message import MIMEMessage
import email
import time
import imaplib
import os
import pickle
import getpass

from dataset import *
from classifier import *
from stemming import *
from scikit_classifier import sci_classify

fromaddr = "EMAIL HERE" 
pwd = getpass.getpass()
imap_server = "outlook.office365.com"
imap_port = 993
smtp_server = "smtp.office365.com"
smtp_port = 587
# TODO: Once bot is ready deploy it by having it search for the XSEDE tickets
# WARNING: Only deploy the bot after more testing, otherwise it could spam users' inboxes.
# search_for = 'tickets.xsede.org'
search_for = 'Test Ticket'

def main():
	M = imaplib.IMAP4_SSL(imap_server, imap_port)
	server = smtplib.SMTP(smtp_server, smtp_port)
	try:
		M.login(fromaddr, pwd)
		server.starttls()
		server.login(fromaddr, pwd)
	except:
		print('Error: Login failed')

	# our classifier
	table_file = open(os.path.join('bin', 'table.p'), 'rb')
	table = pickle.load(table_file)
	table_file.close()

	# bayes classifier
	# clf_file = open(os.path.join('bin', 'clf.p'), 'rb')
	# clf = pickle.load(clf_file)
	# clf_file.close()
	# master_dict = read_dictionary(os.path.join('bin', 'dictionary.txt'))
	# words = master_dict.keys()

	# stopwords
	stopwords = []
	with open(os.path.join(os.getcwd(), 'bin', 'stopwords.txt'), 'r') as stop_file:
		stopwords += stop_file.read().split()

	# responder
	reply_to_emails(M, server, table, stopwords)
	# reply_to_emails(M, server, table, stopwords, clf, words)
	server.quit()

def get_text(msg):
	if msg.is_multipart():
		return get_text(msg.get_payload(0))
	else:
		return msg.get_payload(None, True)

def strip_one_text(text, stopwords):
	# remove all threads (lines that start with '>')
	not_threads = []
	for l in text.split('\n'):
		if not is_thread(l):
			not_threads.append(l)
	text = ' '.join(not_threads)
	# res[0] = info from secondary header, res[1] = body text after removing secondary header
	sec_header_res = get_sec_header_info(text)
	header_info_dict = sec_header_res[0]
	#remove the secondary header
	text = sec_header_res[1]
	if not_threads and len(header_info_dict) > 0:
		#remove the body header
		ticket_regex = re.compile(r'Ticket.*>')
		#delete everything before the url
		mo = ticket_regex.search(text)
		if mo:
			text = text[text.index(mo.group()) + len(mo.group()):]
		words = text.split()
		# put everything in lowercase
		words = [w.lower() for w in words]
		# remove urls
		words = [re.sub(r'((http://|https://)?([a-z0-9-_]+\.)+([a-z]{2,4})(/[a-z0-9-_#]+)*)(\.[a-z]{3})?', '', w) for w in words]
		#remove file paths
		words = [re.sub(r'(/\w)+\.\w+', '', w) for w in words]
		#remove escape characters (i.e. \n, \r)
		words = [re.sub(r'\\\w', ' ', w) for w in words]
		#removes all digits
		words = [re.sub(r'\d','',w) for w in words]
		# remove punctuation
		words = [re.sub(r'\W',' ',w) for w in words]
		# split each word by spaces and other separators
		words = [w2 for w in words for w2 in re.split(r'\s|_|-|/|\(|\)|\.', w)]
		#add the word (w) to the list (words) if the w isn't one of the stop words
		words = [stem(w.strip()) for w in words if stem(w.strip()) not in stopwords and len(w) > 1]
		#remove name in words and add category to words
		if 'Ticket created' in header_info_dict:
			user = header_info_dict['Ticket created'].lower()
			words = [w for w in words if w != stem(user)]
		if 'From' in header_info_dict:
			name = header_info_dict['From'].split()
			name = [stem(n.lower()) for n in name]
			words = [w for w in words if w not in name]
		if 'Category' in header_info_dict:
			category = re.split(r'\s|/', header_info_dict['Category'])
			category = [c.lower() for c in category if c != 'Other']
			for c in category:
				words.append(stem(c.lower()))
		#join the list together with spaces in between
		text = ' '.join(words)
	else:
		text = ''
	return text

def write_one_dictionary(body):
	word_dict = {}
	words = body.split()
	words = [word.lower() for word in words]
	for word in words:
		# increment local and global dicts
		word_dict.setdefault(word, 0)
		word_dict[word] += 1
	return word_dict

def send_reply(server, text, original=None):
	msg = MIMEMultipart('mixed')
	body = MIMEMultipart('alternative')

	if original is not None:
		text = append_original_text(text, original)
		# Fix subject
		msg["Subject"] = "RE: " + original["Subject"].replace("Re: ", "").replace("RE: ", "")
		msg['In-Reply-To'] = original["Message-ID"]
		msg['References'] = original["Message-ID"]
		msg['Thread-Topic'] = original["Thread-Topic"]
		msg['Thread-Index'] = original["Thread-Index"]
		
	toaddr = original['From']
	body.attach(MIMEText(text, 'plain'))
	msg.attach(body)
	server.sendmail(fromaddr, toaddr, msg.as_string())

def append_original_text(text, original):
    newhtml = ""
    newtext = ""

    for part in original.walk():
        if (part.get('Content-Disposition')
            and part.get('Content-Disposition').startswith("attachment")):

            part.set_type("text/plain")

            del part["Content-Disposition"]
            del part["Content-Transfer-Encoding"]

        if part.get_content_type().startswith("text/plain"):
        	newtext += "\n"
        	newtext += part.get_payload(decode=False)

    return text+'\n\n'+newtext
	
def reply_to_emails(M, server, table, stopwords, clf=None, words=None):
	'''
	For some reason the bot ignores the queries "UNSEEN" and "UNANSERED". 
	The workaround is to maintain a .txt file called "replied_to,txt"
	that stores the "num" values of the emails that it has already replied to,
	'''
	rv, data = M.select('INBOX', readonly=True)
	if rv == 'OK':
		query = '(UNSEEN UNANSWERED UNDELETED SINCE "01-Aug-2018"' + ' NOT FROM "' + fromaddr + '" HEADER SUBJECT "' + search_for + '")'
		rv, data = M.search(None, query)
		if rv != 'OK':
			sys.exit(1)
		nums = data[0].split()
		replied_to = []
		replied_to_path = os.path.join(os.getcwd(), 'bin', 'replied_to.txt')
		if not os.path.isfile(replied_to_path):
			_ = open(replied_to_path, 'w')
		with open(replied_to_path, 'r') as replied_to_file:
			replied_to = replied_to_file.read().split()
		for num in nums:
			if num.decode() in replied_to:
				continue 
			try:
				rv, data = M.fetch(num, '(RFC822)')
				original = email.message_from_bytes(data[0][1])
				original_subject = original['Subject']
				original_body = get_text(original).decode()
				word_dict = write_one_dictionary(strip_one_text(original_body, stopwords))
				# TODO: if word_dict is empty the classifier won't be able to classify the email, once deployed the bot should ignore this email.
				# if not word_dict:
				# 	continue
				prediction, confidence = classify(word_dict, table)
				# prediction, confidence = sci_classify(word_dict, words, clf)
				reply_body = ''
				sent_reply = True
				if confidence > 0.5:
					# TODO: improve bot's user feedback to actually be helptful.
					reply_body = 'Hello,\n\nThis is a bot. I\'ve classified your issue as: ' + prediction + '. Hope this helped!\n\nLove,\n\nbot'
				else:
					# TODO: Once deployed the bot should not send a reply if it's not confident in its classification.
					# continue
					# sent_reply = False
					reply_body = 'Hello\n\nThis is a bot. Sorry, I can\'t seem to identify your issue!\n\nLove,\n\nbot'
				if sent_reply:
					print('sent reply to: ' + original['Subject'] + ' from ' + original['From'])
				send_reply(server, reply_body, original)
			except ConnectionResetError:
				continue
			replied_to.append(num.decode())
		with open(replied_to_path, 'w') as replied_to_file:
			replied_to_file.write('\n'.join(replied_to))

if __name__ == '__main__':
	main()
