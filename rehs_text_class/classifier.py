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
import string
import logging
import ast
import random
import pickle

import flags
from dataset import toggle_tqdm, read_dictionary, get_first_uids
from table import Table
from collections import Counter

# Uses local dictionary/vector of each email to fill in table
def build_table(path, exclusionary_list=[], save=True):
	# make a new table object
	table = Table() 
	# open the dictonary of words
	word_dict = read_dictionary(os.path.join('bin', 'dictionary.txt')) 
	for word in word_dict.keys():
		# add a column for each word
		table.add_column(word) 
	# loop through the first uids
	for uid in get_first_uids(os.path.join(path, 'training_emails')): 
		if uid not in exclusionary_list:
			# if the uid directory has a body_stripped file and a local dictionary/vector file
			if os.path.isfile(os.path.join(uid, 'body_stripped.txt')) and os.path.isfile(os.path.join(uid, 'dictionary.txt')):
				word_dict = open(os.path.join(uid, 'dictionary.txt'))
				# get the local dictionary/vector
				vector = ast.literal_eval(word_dict.read())
				word_dict.close()
				# get all the titles from the table (the words from the dictonary)
				titles = table.get_titles()
				new_row = []
				# loop through the titles of the table
				for title in titles: 
					count = 0
					if title in vector:
						# set count to equal the word frequency in vector
						count = vector[title] 
					# add count to the new_row
					new_row.append(count) 
				#add the new_row to the table
				#get the name of the parent's parent directory, aka the category that the ticket was placed in
				catg_name = os.path.basename(os.path.abspath(os.path.join(uid, os.pardir, os.pardir)))
				# if the category is not just the emails directory, classify it as it's category
				if catg_name != 'emails':
					table.add_row(new_row, catg_name, uid)
	if save:
		pickle.dump(table, open(os.path.join('bin', 'table.p'),'wb') )
		logging.info('wrote new table to file')
	return table
	
# returns a category given the local dictionary of an email
def classify(word_dict, table):
	# a dictionary of classifiers paired with the number of emails with that classification
	classifiers = Counter(table.get_classifiers())
	class_percents = {}
	# loop through the classifiers
	for key in classifiers.keys():
		# set the class_percents[key] to the percent of total emails that have the given classification
		class_percents[key] = classifiers[key]/len(table.get_row_labels())
	word_percents = {}
	# loop through the classifiers
	for key in class_percents.keys(): 
		# a list of table indices with the given classification
		indices = table.get_classifier_indices(key)
		total = 0
		# loop through the indices
		for index in indices: 
			# add the number of words at the given row to the total
			total = total + sum(table.get_row(index))
		percents = []
		# loop through every word in table
		for word in table.get_titles():
			# keep track of the total occurances a word is used in an email with a given classification
			word_total = 0
			# get the index of the given word from the table titles
			word_index = table.get_titles().index(word)
			for index in indices:
				# add to word_total with the number in a certain cell on the table
				word_total = word_total + table.get_from_pos(word_index, index)
			# add the percent for this word to the list of percents (the equation for finding the percent is (n_k+1)/(n+vocabulary))
			percents.append((word_total+1)/(total+len(table.get_titles())))
		# add the list of percents to the dictonary
		word_percents[key] = percents
	class_probabilities = {}
	# loop through classifiers
	for classifier in class_percents.keys():
		# probability = 1
		# initially set the probability to the probability of a given classifier (previously calculated)
		probability = class_percents[classifier]
		 # loop through the words in the table
		for word, freq in word_dict.items():
			# if the word is in the email:
			if word in table.get_titles():
				index = table.get_titles().index(word)
				# given the classifier, multiply the probability by the probability of the word raised to the power of its frequency
				probability *= (word_percents[classifier][index] ** freq)
		class_probabilities[classifier] = probability
	# add all of the probabilities together
	total = sum(class_probabilities.values()) 
	if total == 0:
		logging.debug(total)
		logging.debug(class_probabilities)
		return 'unknown', 1
	for k, v in class_probabilities.items():
		class_probabilities[k] /= total
	# return the category with the highest probability
	max_key = max(class_probabilities, key=class_probabilities.get)
	return max_key, class_probabilities[max_key]

# classifies tickets, moving them from emails to a category in test_emails
def auto_classify(emails_path, test_path):
	if not os.path.isfile(os.path.join('bin', 'table.p')):
		print('Error: no table found, run tble to build table')
		return
	# open the table
	table_file = open(os.path.join('bin', 'table.p'), 'rb')
	table = pickle.load(table_file)
	table_file.close()
	for ticket_id in toggle_tqdm(os.listdir(emails_path)):
		if ticket_id == 'extra':
			continue
		ticket_path = os.path.join(emails_path, ticket_id)
		# loop over the first uid of each ticket (for loop only runs once)
		for uid_path in get_first_uids(ticket_path):
			dict_file = os.path.join(uid_path, 'dictionary.txt')
			if not os.path.isfile(dict_file):
				continue
			word_dict = read_dictionary(dict_file)
			# use the body_stripped file to categorize the email
			category, probability = classify(word_dict, table)
			if category == 'unknown':
				logging.info('The classifier could not determine a category.')
				continue
			logging.info(probability)
			# move the email to its respective category in test_emails
			if not category in os.listdir(test_path):
				os.makedirs(os.path.join(test_path, category))
			os.rename(os.path.join(emails_path, ticket_path), os.path.join(test_path, category, ticket_id))

# transfers emails from test_emails to training_emails under the supervision of a human
def manual_check(training_path, test_path):
	# open the table (this is just for listing the categories)
	if not os.path.isfile(os.path.join('bin', 'table.p')):
		print('Error: no table found, run tble to build table and clss to auto-classify emails')
		return
	table_file = open(os.path.join('bin', 'table.p'), 'rb')
	table = pickle.load(table_file)
	table_file.close()
	# a list of categories to show the user
	categories_string = 'avaliable categories:'
	for c in list(Counter(table.get_classifiers()).keys()):
		categories_string += '\n' + c
	# iterate over each category directory in test_emails
	for category in os.listdir(test_path):
		if not os.path.isdir(os.path.join(test_path, category)):
			continue
		for ticket_id in os.listdir(os.path.join(test_path, category)):
			os.system('cls' if os.name == 'nt' else 'clear')
			ticket_path = os.path.join(test_path, category, ticket_id)
			for uid_path in get_first_uids(ticket_path):
				body_file = os.path.join(uid_path, 'body.txt')
				if not os.path.isfile(body_file):
					continue
				with open(body_file) as body:
					# print the email body to the user
					print(body.read())
					user_input = ''
					while user_input != 'y' and user_input != 'n':
						# ask user if email was classified to the right category
						user_input = input('does this email belong in category ' + category + '? (y or n): ')
					# user answered no, we must move the email to the proper category in training_emails
					if user_input == 'n':
						unknown = False
						category_input = 'BLANK'
						print(categories_string)
						# if user answers no, prompt user for the proper category (note that the proper category must be in training_emails)
						while not os.path.isdir(os.path.join(training_path, category_input)):
							category_input = input('what category does this email belong to?: (type \'none\' to place back in emails) ')
							if category_input == 'none':
								# if user does not know category, put email in extra (can be undone with ucat)
								os.rename(ticket_path, os.path.join('emails', 'extra', ticket_id))
								unknown = True
								break
						if not unknown:
							# move email to the proper category
							os.rename(ticket_path, os.path.join(training_path, category_input, ticket_id))
							#append ticket number to proper category .txt file
							ticket_num = re.sub(r'\D', '', ticket_id)
							catg_file = open(os.path.join(os.getcwd(), os.path.join('categories', category+'.txt')),'a+')
							catg_file.write("\n"+ticket_num)
					# user answered yes, we can move the email to its respective category in training_emails
					else:
						os.rename(ticket_path, os.path.join(training_path, category, ticket_id))
						 #append ticket number to proper category .txt file
						ticket_num = re.sub(r'\D', '', ticket_id)
						catg_file = open(os.path.join(os.getcwd(), os.path.join('categories', category+'.txt')),'a+')
						catg_file.write("\n"+ticket_num)

# runs an accuracy test on the currently trained table using the emails of the testing_emails_list
def test_accuracy(testing_emails_list, table=None):
	if not table:
		table_file = open(os.path.join('bin', 'table.p'), 'rb')
		table = pickle.load(table_file)
		table_file.close()
	# open input file
	with open(testing_emails_list, 'r') as test_emails:
		correct = 0
		total = 0
		output = []
		for uid in toggle_tqdm(test_emails.read().split()):
			dict_path = os.path.join(uid, 'dictionary.txt')
			if os.path.isdir(uid) and os.path.isfile(dict_path):
				# open dictionary
				word_dict = read_dictionary(dict_path)
				# get category from parent's parent's directory name (the name of the category directory)
				category = os.path.basename(os.path.abspath(os.path.join(uid, os.pardir, os.pardir)))
				# classify using the dictionary
				guess, confidence = classify(word_dict, table)
				# build output
				out = (uid, guess, str(confidence*100)[:4], '\33[41m', category, '\33[0m')
				if guess == category:
					out = (uid, guess, str(confidence*100)[:4], '\33[42m', category, '\33[0m')
					correct += 1
				total += 1
				output.append(out)
			else:
				logging.info("Unable to use " + uid)
		# final score, grading output
		return correct/total, output