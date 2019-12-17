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
import os
import sys
import getpass

import flags
from dataset import *
from classifier import *
from visual import *
from scikit_classifier import *

def main():
	# initialize flags
	flags.init()
	# process command modifiers
	for i in range(2, len(sys.argv)):
		flag = sys.argv[i]
		if flag == '-vb':
			flags.verbose = True
		elif flag == '-nl':
			flags.load_bar = False
		elif flag == '-re':
			flags.redo = True
		elif flag == '-dl':
			flags.smod_add = False
		elif flag == '-vi':
			flags.make_visual = True

	# enable/disable verbose output
	logging.basicConfig(level=logging.INFO, format=' %(message)s')
	if not flags.verbose:
		logging.disable(logging.INFO) # comment out to enable output

	# determine which command to run
	mode = sys.argv[1]
	# dwnl: downloads emails and organizes them into directories
	if mode == 'dwnl':
		# log in to office account
		M = imaplib.IMAP4_SSL('outlook.office365.com', 993)
		logging.info('\n'.join(i for i in M.capabilities if i.startswith("AUTH=")))
		try:
			M.login('mkandes@ucsd.edu', getpass.getpass())
			rv, mailboxes = M.list()
			if rv == 'OK':
				logging.info('Mailboxes: ', mailboxes)
			rv, data = M.select('INBOX', readonly=True)
			if rv == 'OK':
				logging.info('Searching mailbox ...')
				# rv, data = M.search(None, '(HEADER FROM "help@xsede.org")')
				rv, data = M.search(None, '(HEADER SUBJECT "tickets.xsede.org")')
				if rv != 'OK':
					logging.info('No messages found!')
					sys.exit(1)
				download_emails(data, M)
			M.close()
		except imaplib.IMAP4.error:
			print ('Error: login failed!')
		# create the categories directory if it does not yet exist
		categories_path = os.path.join(os.getcwd(), 'categories')
		if not os.path.isdir(categories_path):
			os.makedirs(categories_path)
		return

	# strp: strips downloaded email body text
	elif mode == 'strp':
		stop_file_paths = [os.path.join(os.getcwd(), 'bin', 'stopwords.txt')]
		emails_path = os.path.join(os.getcwd(), 'emails')
		if not os.path.isdir(emails_path):
			print('Error: no emails directory was found, run python3 main.py dwnl to download emails')
			return
		if len(sys.argv) >= 3:
			stop_file_paths += [os.path.join(os.getcwd(), 'bin', arg) for arg in sys.argv[2:] if arg[-4:] == '.txt']
		stopwords = []
		for stop_file_path in stop_file_paths:
			if not os.path.isfile(stop_file_path):
				print('Error: ' + stop_file_path + ' was not found')
				return
			with open(stop_file_path, 'r') as stop_file:
				stopwords += stop_file.read().split()
		logging.info('Stripped:')
		extra_path = os.path.join(os.getcwd(), os.path.join('categories', 'extra.txt'))
		with open(extra_path, 'a+') as extra_file:
			for path in toggle_tqdm(get_first_uids(emails_path)):
				if strip_text(path, stopwords, extra_file):
					logging.info(path)
		return

	# dict: writes a dictionary file containing the frequency of each word
	elif mode == 'dict':
		emails_path = os.path.join(os.getcwd(), 'emails')
		if not os.path.isdir(emails_path):
			print('Error: no emails directory was found, run python3 main.py dwnl and python3 main.py strp')
			return
		write_dictionary(os.getcwd())
		return

	# catg [FILENAME.txt]: Moves ticket directories specified by FILENAME.txt
	# into a category directory named FILENAME.
	elif mode == 'catg':
		# path to emails directory
		emails_path = os.path.join(os.getcwd(), 'emails')
		if not os.path.isdir(emails_path):
			print('Error: no emails directory was found, run python3 main.py dwnl to download emails')
			return
		# path to categories directory
		categories_path = os.path.join(os.getcwd(), 'categories')
		if not os.path.isdir(categories_path):
			print('Error: no categories directory was found, create this directory and put input file inside of it')
			return
		# path to training emails directory
		training_path = os.path.join(os.getcwd(), 'training_emails')
		if not os.path.isdir(training_path):
			os.makedirs(training_path)
		categorize_tickets(emails_path, categories_path, training_path)
		return

	# tble: creates a table using dictionaries
	elif mode == 'tble':
		emails_path = os.path.join(os.getcwd(), 'emails')
		dict_path = os.path.join('bin', 'dictionary.txt')
		if not os.path.isfile(dict_path) or not os.path.isdir(emails_path):
			print('Error: dictionary.txt was not found in this directory, run main.py with dwnl, strp, and then dict')
		table = build_table(os.getcwd()) #create a new table
		return

	# clss: automatically categorize emails in emails directory using a prototype Naive Bayes classifier
	elif mode == 'clss':
		emails_path = os.path.join(os.getcwd(), 'emails')
		if not os.path.isdir(emails_path):
			print('Error: no emails directory found, run main.py with dwnl')
			return
		test_path = os.path.join(os.getcwd(), 'test_emails')
		if not os.path.isdir(test_path):
			os.makedirs(test_path)
		auto_classify(emails_path, test_path)
		return

	# manc: manually approve automatic categorization of clss and transfer emails from test_emails to training_emails
	elif mode == 'manc':
		training_path = os.path.join(os.getcwd(), 'training_emails')
		if not os.path.isdir(training_path):
			print('Error: no training data avaliable, run catg command')
			return
		test_path = os.path.join(os.getcwd(), 'test_emails')
		if not os.path.isdir(test_path):
			print('Error: test_emails directory is empty, run clss command')
			return
		manual_check(training_path, test_path)

	# ucat: uncategorize emails from test_emails and training_emails
	elif mode == 'ucat':
		emails_path = os.path.join(os.getcwd(), 'emails')
		if not os.path.isdir(emails_path):
			print('Error: no emails directory found, run dwnl command')
			return
		user_input = ''
		while (user_input != 'y' and user_input != 'n'):
			user_input = input('WARNING: ucat command will uncategorize every email, are you sure you want to proceed? (y, n): ')
		if (user_input == 'y'):
			training_path = os.path.join(os.getcwd(), 'training_emails')
			if os.path.isdir(training_path):
				uncategorize_tickets(training_path)
			test_path = os.path.join(os.getcwd(), 'test_emails')
			if os.path.isdir(test_path):
				uncategorize_tickets(test_path)
			extra_path = os.path.join(os.getcwd(), 'emails', 'extra')
			if os.path.isdir(extra_path):
				for ticket_id in os.listdir(extra_path):
					os.rename(os.path.join(extra_path, ticket_id), os.path.join(os.getcwd(), 'emails', ticket_id))

	# smod [FILENAME.txt]: adds words to stopwords.txt, which is used to strip
	elif mode == 'smod':
		if len(sys.argv) < 3:
			print('Error: smod must be succeeded by an input file, Usage: python main.py smod FILENAME.txt')
			return
		file_name = sys.argv[sys.argv.index('smod') + 1]
		stop_path = os.path.join('bin', 'stopwords.txt')
		if not os.path.isfile(stop_path):
			print('Error: stopwords.txt is missing from this directory')
			return
		elif file_name[-4:] == '.txt' and os.path.isfile(os.path.join('bin', file_name)):
			with open(os.path.join('bin', file_name), 'r') as text_file:
				new_words = text_file.read().split()
				curr_words = open(stop_path).read().split()
				for w in new_words:
					if flags.smod_add:
						if w not in curr_words:
							curr_words.append(w)
						else:
							print(w + ' is already in stopwords.txt')
					else:
						if w in curr_words:
							curr_words.remove(w)
						else:
							print(w + ' was not found in stopwords.txt')
				with open(stop_path, 'w') as stop_file:
					stop_file.write('\n'.join(curr_words))
		else:
			print('Error: no valid input file was provided')
		return

	# list: selects some emails to test against the algorithm
	elif mode == 'list':
		percent = 0
		if len(sys.argv) >= 3 and sys.argv[2].isdigit():
			percent = int(sys.argv[2])
		else:
			percent = 10
		uids = get_first_uids(os.path.join(os.getcwd(), 'training_emails'))
		with open(os.path.join('bin', 'testing_list.txt'), 'w+') as testing_list:
			random.shuffle(uids)
			testing_uids = uids[:int(len(uids) * percent / 100)]
			for uid in testing_uids:
				testing_list.write(os.path.relpath(uid) + '\n')
			build_table(os.getcwd(), exclusionary_list=testing_uids)
		return

	# acrc: tests accuracy of algorithm using test emails chosen in list
	elif mode == 'acrc':
		accuracy, output = test_accuracy(os.path.join('bin', 'testing_list.txt'))
		for line in output:
			print('{}: {}, {}% {}(Correct: {}){}'.format(line[0], line[1], line[2], line[3], line[4], line[5]))
		print("Naive Bayes model accuracy: {}%".format(str(accuracy*100)))
		return

	elif mode == 'crsv':
		num_tests = 0
		if len(sys.argv) >= 3 and sys.argv[2].isdigit():
			num_tests = int(sys.argv[2])
		else:
			num_tests = 4
		print('Running {}-fold cross validation...'.format(num_tests))
		testing_lists = []
		#get all first uids and randomly assign them to lists
		uids = get_first_uids(os.path.join(os.getcwd(), 'training_emails'))
		random.shuffle(uids)
		#create a list of testing lists
		testing_lists = [uids[int(len(uids)*(1/num_tests*i)):int(len(uids)*(1/num_tests*(i+1)))] for i in range(num_tests)]

		total_accuracy = 0
		outputs = []
		#loop through the testing lists
		for list in testing_lists:
			#build a table, excluding the list
			table = build_table(os.getcwd(), save=False, exclusionary_list=list)
			#save the list to a file
			with open(os.path.join('bin', 'temp_cross_validation.txt'), 'w+') as test_file:
				for uid in list:
					test_file.write(os.path.relpath(uid) + '\n')
			#get the accuracy and output from testing the list
			accuracy, output = test_accuracy(os.path.join('bin', 'temp_cross_validation.txt'), table=table)
			total_accuracy += accuracy
			outputs += output
		#delete the temporary file
		os.remove(os.path.join('bin', 'temp_cross_validation.txt'))

		#average the accuracies
		total_accuracy /= num_tests
		actual = []
		predicted = []
		for line in outputs:
			actual.append(line[4])
			predicted.append(line[1])
			print('{}: {}, {}% {}(Correct: {}){}'.format(line[0], line[1], line[2], line[3], line[4], line[5]))
			# print(line)
		if flags.make_visual:
			C = confusion_matrix(actual, predicted)
			# C = np.delete(C, (16,), axis=0)
			# C = np.delete(C, (16,), axis=1)
			print(C.shape)
			class_names = []
			with open(os.path.join('bin', 'table.p'), 'rb') as table_file:
				table = pickle.load(table_file)
				class_names = np.unique(table.get_classifiers())
			# Plot normalized confusion matrix
			plot_confusion_matrix(C, total_accuracy*100, classes=class_names, normalize=True, title='Our Bayes Normalized confusion matrix')
			if not os.path.isdir('graphs'):
				os.makedirs('graphs')
			plt.savefig('graphs/our_confusion_matrix.png')
			plt.show()

		print("Naive Bayes model {}-fold cross validation: {}%".format(num_tests, str(total_accuracy*100)))
		return

	elif mode == 'bcsv':
		with open(os.path.join('bin', 'table.p'), 'rb') as table_file:
			table = pickle.load(table_file)
			build_csv(os.path.join(os.getcwd(), 'bin', 'training_data.csv'), os.path.join(os.getcwd(), 'bin', 'testing_data.csv'), table)
			if flags.make_visual:
				for category_file in os.listdir(os.path.join(os.getcwd(), 'categories')):
					build_pie_chart(category_file[:-4], os.path.join(os.getcwd(), 'bin', 'training_data.csv'))

	# htxt: access help text
	elif mode == 'htxt':
		help_path = os.path.join('bin', 'help.txt')
		if os.path.isfile(help_path):
			with open(help_path) as file:
				print(file.read())
		else:
			print('Error: help.txt was not found in this directory')
		return

	elif sci_mode(mode):
		logging.info('Performed a scikit command')
	
	else:
		print('Error: ' + mode + ' is not a valid command')
		return

if __name__ == '__main__':
	main()
