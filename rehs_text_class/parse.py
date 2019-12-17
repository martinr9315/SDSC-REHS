#!/usr/bin/env python3

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
'''
import re
import email
import string

def get_uid(msg_data):
	UID = msg_data[0].decode()
	UIDRegex = re.compile(r'(UID )(\d+)')
	return(UIDRegex.search(UID).group(2))

def get_ticket_id(text):
	id_regex = re.compile(r'(Request )(\d+)')
	mo = id_regex.search(str(text))
	ticketID = mo.group(2)
	return (ticketID)

def get_emails(text):
	emailRegex = re.compile(r'''(
		[a-z0-9._%+-]+
		@
		[a-zA-Z0-9._]+
		(\.[a-zA-z]{2,4})
		)''', re.VERBOSE)

	email_list = []
	for groups in emailRegex.findall(text):
		if groups[0] not in email_list:
			email_list.append(groups[0])

	return email_list

def get_phone_numbers(text):
	phoneRegex = re.compile(r'''(
		(\d{3}|\(\d{3}\))?                # area code
		(\s|-|\.)?                        # separator
		(\d{3})                           # first 3 digits
		(\s|-|\.)                         # separator
		(\d{4})                           # last 4 digits
		(\s*(ext|x|ext.)\s*(\d{2,5}))?    # extension
		)''', re.VERBOSE)

	phone_list = []
	for groups in phoneRegex.findall(text):
		phoneNum = '-'.join([groups[1], groups[3], groups[5]])
		if groups[8] != '':
			phoneNum += ' x' + groups[8]
		if phoneNum not in phone_list:
			phone_list.append(phoneNum)

	return phone_list

def get_sec_header_info(text):
	header_regex_list = (
	re.compile(r'\[(Ticket created) from .*? by (.*?)\]'),
	re.compile(r'\[(From): (.*?)\]'),
	re.compile(r'\[(System): (.*?)\]'),
	re.compile(r'\[(Category): (.*?)\]'))
	header_info_dict = {}
	for r in header_regex_list:
		mo = r.search(text)
		text = re.sub(r,'', text)
		if mo:
			header_info_dict[mo.group(1)] = mo.group(2)
	return (header_info_dict, text)

def get_file_paths(text):
	words = text.split()
	file_regex = re.compile('(/\w+)+\.\w+')
	paths = file_regex.findall(text)
	new_words = []
	for w in words:
		word = re.sub('(/\w+)+\.\w+', '', w)
		new_words.append(word)
	return ' '.join(new_words)

def get_ticket_from_subject(text):
	idRegex = re.compile(r'(\[)(tickets\.xsede\.org #)(\d+)(\])')
	mo = idRegex.search(str(text))
	if not mo:
		return None
	ticketID = mo.group(3)
	return (ticketID)

def get_ticket_from_body(text):
	idRegex = re.compile(r'(Request )(\d+)')
	mo = idRegex.search(str(text))
	if not mo:
		return None
	ticketID = mo.group(2)
	return (ticketID)