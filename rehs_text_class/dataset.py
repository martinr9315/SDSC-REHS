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
import os
import imaplib
import email
import string
import logging
import json
import ast
from tqdm import tqdm

import flags
from stemming import *
from parse import *

def download_emails(data, M):
	#make emails directory
	email_path = os.path.join('.', 'emails')
	try:
		os.makedirs(email_path)
	except os.error:
		logging.info('directory ' + email_path + ' was already created')

	nums = data[0].split()
	# splice nums up to where we last downloaded
	if not flags.redo and os.path.isfile(os.path.join('bin', 'last_num.txt')):
		with open(os.path.join('bin', 'last_num.txt'), 'r') as last_num_file:
			read_value = last_num_file.readline()
			if read_value != '':
				while not read_value.encode() in nums:
					read_value = str(int(read_value) - 1)
				nums = nums[nums.index(read_value.encode()):]
				logging.info('successfully spliced list')
	last_num = ''
	emails_path = os.path.join('.', 'emails')
	training_path = os.path.join('.', 'training_emails')
	test_path = os.path.join('.', 'test_emails')
	for num in toggle_tqdm(nums):
		# get ticket ID to categorize UID directories
		try:
			rv, data = M.fetch(num, '(RFC822)')
			msg = email.message_from_string(data[0][1].decode())
			# try getting ticket id from the subject (faster than from the body)
			ticketID = get_ticket_from_subject(msg['Subject'])
			if not ticketID:
				# ticket ID was not found in the subject, try finding it in the body
				ticketID = get_ticket_from_body(msg)
				# ticket ID was not found in the subject or the body, give up
				if not ticketID:
					continue
			#create new ticket directory
			ticket_name = 'ticket#' + str(ticketID)
			logging.info(ticket_name)
			# don't create the ticket directory if it exists (even in a category file)
			found_dir_rv = found_directory(ticket_name, emails_path) or found_directory(ticket_name, training_path) or found_directory(ticket_name, test_path)
			ticket_path = ''
			if not found_dir_rv:
				ticket_path = os.path.join(emails_path, ticket_name)
				os.makedirs(ticket_path)
			else:
				ticket_path = found_dir_rv
				logging.info('directory ' + ticketID + ' was already created')
				if not flags.redo:
					last_num = num
					continue
			# get UID to name directory
			rv, msg_data = M.fetch(num, 'UID')
			if rv != 'OK':
				logging.info('ERROR getting message data', num)
				exit()
			UID = get_uid(msg_data)
			UID_path = os.path.join(ticket_path, 'UID#' + str(UID))

			# make new directory by UID for each email
			if not os.path.isdir(UID_path):
				os.makedirs(UID_path)
			else:
				logging.info('directory ' + UID_path + ' was already created')

			# writing the header file
			rv, msg_data = M.fetch(num, '(BODY.PEEK[HEADER])')
			header = open(os.path.join(UID_path, "header.txt"), "w+")
			for response_part in msg_data:
				if isinstance(response_part, tuple):
					header.write(response_part[1].decode())
			header.close()

			# writing the body file
			rv, msg_data = M.fetch(num, '(BODY.PEEK[TEXT])')
			body = open(os.path.join(UID_path, "body.txt"), "w+")
			for response_part in msg_data:
				if isinstance(response_part, tuple):
					body.write(response_part[1].decode())
			body.close()

			# writing the subject file
			rv, data = M.fetch(num, '(RFC822)')
			msg = email.message_from_string(data[0][1].decode())
			subject = open(os.path.join(UID_path, "subject.txt"), "w+")
			subject.write(msg['Subject'])
			subject.close()
			logging.info('Finished Writing ' + str(UID_path))

			# keep track of most recently downloaded emails
			last_num = num
		except ConnectionResetError:
			print ('Error: ConnectionResetError')
			continue
		except KeyboardInterrupt:
			# save the last_num before quitting
			with open(os.path.join('bin', 'last_num.txt'), 'w') as last_num_file:
				if last_num:
					last_num_file.write(last_num.decode())
			exit()
	# save the most recently downloaded ticket
	if last_num:
		with open(os.path.join('bin', 'last_num.txt'), 'w') as last_num_file:
			last_num_file.write(last_num.decode())

def strip_text(uidpath, stopwords, extra_file):
	if os.path.isfile(os.path.join(uidpath, 'body_stripped.txt')) and not flags.redo:
		return False
	#open the body text file
	if not os.path.isfile(os.path.join(os.getcwd(), os.path.join(uidpath, 'body.txt'))):
		return False
	file = open(os.path.join(os.getcwd(), os.path.join(uidpath, 'body.txt')))
	text = file.read()
	file.close()
	text2 = text
	# check if email is actually the first of its ticket by looking for threads: lines starting with one or more '>'
	# a lack of a secondary header is tolerable.
	has_threads = False
	for l in text.split('\n'):
		if is_thread(l):
			has_threads = True
			break
	if not has_threads:
		# extract info from body
		email_list = get_emails(text)
		phone_list = get_phone_numbers(text)
		# res[0] = info from secondary header, res[1] = body text after removing secondary header
		sec_header_res = get_sec_header_info(text)
		header_info_dict = sec_header_res[0]
		# writing the info file
		info = open(os.path.join(os.getcwd(), os.path.join(uidpath, 'info.txt')), "w+")
		info.write('Email list: ' + str(email_list) + '\n' + 'Phone number list: ' + str(phone_list) + '\n' + 'Header: ' + str(header_info_dict))
		info.close()
		#remove the secondary header
		text = sec_header_res[1]
		# if not_threads and len(header_info_dict) > 0:
		#remove the body header
		ticket_regex = re.compile(r'Ticket <URL:.*>')
		#delete everything before the url
		mo = ticket_regex.search(text)
		if mo:
			text = text[text.index(mo.group()) + len(mo.group()):]
		else:
			#write ticket numbers not being stripped to extra.txt
			ticket_num = os.path.basename(os.path.abspath(os.path.join(uidpath, os.pardir)))
			ticket_num = re.sub(r'\D', '', ticket_num)
			extra_file.write(ticket_num + '\n')
			return False
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
		#add the stemmed word (w) to the list (words) if the stemmed w isn't one of the stop words
		words = [stem(w.strip()) for w in words if stem(w.strip()) not in stopwords and len(w) > 1]
		#create a 'body_stripped' text file in the same directory
		body_stripped = open(os.path.join(uidpath, 'body_stripped.txt'), 'w+')
		#remove name and username from words and add category to words
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
		body_stripped.write(' '.join(words))
		body_stripped.close()
		return True
	else:
		#write ticket numbers not being stripped to extra.txt
		ticket_num = os.path.basename(os.path.abspath(os.path.join(uidpath, os.pardir)))
		ticket_num = re.sub(r'\D', '', ticket_num)
		extra_file.write(ticket_num + '\n')
		return False

# path: absolute path to emails directory
def categorize_tickets(emails_path, categories_path, training_path):
	for f in os.listdir(categories_path):
		lost = 0
		if os.path.isfile(os.path.join(categories_path, f)) and f[-4:] == '.txt':
			catg_name = f[:-4]
			# open input file
			with open(os.path.join(categories_path, f), 'r') as tickets_file:
				# get ticket ids from input file (ex. 90569)
				ticket_ids = tickets_file.read().split()
				if catg_name == 'extra':
					extra_path = os.path.join(os.getcwd(), 'emails', 'extra')
					if not os.path.isdir(extra_path):
						os.makedirs(extra_path)
					for ticket_id in ticket_ids:
						ticket_path = os.path.join(emails_path, 'ticket#' + ticket_id)
						if os.path.isdir(ticket_path):
							os.rename(ticket_path, os.path.join(extra_path, 'ticket#' + ticket_id))
				else:
					# if category directory does not exist, create it
					if not os.path.isdir(os.path.join(training_path, catg_name)):
						os.makedirs(os.path.join(training_path, catg_name))
					# loop over ticket_ids and move them into category directory
					for ticket_id in ticket_ids:
						ticket_path = os.path.join(emails_path, 'ticket#' + ticket_id)
						# we don't want 'extra' emails to be treated as training data
						if os.path.isdir(ticket_path):
							os.rename(ticket_path, os.path.join(training_path, catg_name, 'ticket#' + ticket_id))
						else:
							# print("Unable to locate " + ticket_path + " in emails")
							lost += 1
					print('Categorized: {} ({} emails not found)'.format(f[:-4], lost))

# uncategorizes test and training emails
def uncategorize_tickets(categories_path):
	for category in os.listdir(categories_path):
		if os.path.isdir(os.path.join(categories_path, category)):
			for ticket_id in os.listdir(os.path.join(categories_path, category)):
				os.rename(os.path.join(categories_path, category, ticket_id), os.path.join(os.getcwd(), 'emails', ticket_id))

# writes local and global dictionaries
def write_dictionary(path):
	# initialize global dictionary, which will store the occurance of every word from every email
	global_word_dict = {}
	for uid in get_first_uids(path):
		# will only create a dict for stripped text
		if os.path.isfile(os.path.join(uid, 'body_stripped.txt')):
			local_word_dict = {}
			# extract words from stripped body, then close file
			body_stripped = open(os.path.join(uid, 'body_stripped.txt'))
			words = body_stripped.read().split()
			body_stripped.close()
			for word in words:
				# increment local and global dicts
				local_word_dict.setdefault(word, 0)
				local_word_dict[word] += 1
				if os.path.basename(os.path.abspath(os.path.join(uid, os.pardir, os.pardir, os.pardir))) == 'training_emails':
					global_word_dict.setdefault(word, 0)
					global_word_dict[word] += 1
			logging.info('Scanned: ' + uid)
			# dump local dictionary into uid directory
			with open((os.path.join(uid, 'dictionary.txt')), 'w') as file:
				file.write(json.dumps(local_word_dict))
	 # create a new dictonary
	new_dict = {}
	for word in global_word_dict:
		# if the occurance of a word is greater than 2
		if global_word_dict[word] > 2:
			# add the word to the new dictionary
			new_dict[word] = global_word_dict[word]
	# dump global dictionary into main directory
	with open(os.path.join('bin', 'dictionary.txt'), 'w') as file:
		logging.info('writing dictionary')
		file.write(json.dumps(new_dict))

def read_dictionary(path):
	with open(path, 'r') as file:
		# read in the dictonary file
		dictionary = file.read()
		file.close()
		 # return the parsed version of the dictonary (str -> dictonary)
		return ast.literal_eval(dictionary)

def get_first_uids(path): #path to email directory
	uids = []
	uid_regex = re.compile(r'(UID#)(\d+)')
	# recursively walk through emails directory with os.walk()
	for dir_path, dir_names, file_names in os.walk(path): 
		possible_uids = {}
		# loop through each directory found
		for dir_name in dir_names: 
			mo = uid_regex.search(dir_name)
			# if we found a UID directory:
			if mo: 
				# store it
				possible_uids[int(mo.group(2))] = dir_name 
		if possible_uids.keys():
			# get only the first uid
			first_uid = possible_uids[min(possible_uids.keys())] 
			# add it to the uids list
			uids.append(os.path.join(dir_path, first_uid)) 
	return uids

def is_thread(line):
	thread_regex = re.compile(r'^(\s*)(> )+')
	response_from_regex = re.compile(r'\[Response From: ((\w)+(\s)?)+\]')
	date_regex = re.compile(r'On (Mon|Tue|Wed|Thu|Fri|Sat|Sun)|(\d{4}-\d{2}-\d{2})')
	if thread_regex.search(line) or response_from_regex.search(line) or date_regex.search(line):
		return True
	else:
		return False

def read_emails_from_path(path):
	tickets = []
	# loop through all of the tickets
	for file in os.listdir(path): 
		# make the file path absolute
		file = os.path.join(path, file) 
		if os.path.isdir(file):
			uids = []
			# loop through uids in the directory
			for uid in os.listdir(file): 
				# set the uid_path to include the uid folder
				uid_path = os.path.join(file, uid) 
				# uid map, format: (name of file: file contents)
				uid_map = {} 
				 # loop through all files in uid directory
				for txt_file in os.listdir(uid_path):
					openfile = open(os.path.join(uid_path, txt_file))
					# add the filename and text in the file to the file_map
					uid_map[os.path.basename(os.path.join(uid_path, txt_file))] = openfile.read() 
					openfile.close()
				# add the uid_map to the uid list
				uids.append(uid_map) 
			# add the uids to the ticket list
			tickets.append(uids) 
	return tickets

def found_directory(d, path):
	# recursively walk through emails directory with os.walk()
	for dir_path, dir_names, file_names in os.walk(path):
		for dir_name in dir_names:
			if dir_name == d:
				return os.path.join(dir_path, d)
	return ''

def toggle_tqdm(iterable):
	if flags.load_bar:
		return tqdm(iterable)
	else:
		return iterable