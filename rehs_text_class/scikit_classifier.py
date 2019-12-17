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
import numpy as np
import pickle
import random

from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

import flags
from dataset import read_dictionary, get_first_uids, toggle_tqdm
from visual import *

def sci_mode(mode):
	# This function is an extension of the main.py command processer.
	if mode == 'stble':
		sci_train(os.path.join(os.getcwd(), 'training_emails'))
		return True

	elif mode == 'sclss':
		emails_path = os.path.join(os.getcwd(), 'emails')
		if not os.path.isdir(emails_path):
			print('Error: no emails directory found, run dwnl command')
			return True
		test_path = os.path.join(os.getcwd(), 'test_emails')
		if not os.path.isdir(test_path):
			os.makedirs(test_path)
		sci_auto_classify(emails_path, test_path)
		return True

	elif mode == 'slist':
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
			sci_train(os.getcwd(), exclusionary_list=testing_uids)
		return True
		
	elif mode == 'sacrc':
		accuracy, output = sci_test_accuracy(os.path.join('bin', 'testing_list.txt'))
		for line in output:
			print('{}: {}, {}% {}(Correct: {}){}'.format(line[0], line[1], line[2], line[3], line[4], line[5]))
		print("Scikit-Learn Naive Bayes model accuracy: {}%".format(str(accuracy*100)))
		return True

	elif mode == 'scrsv':
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
		# loop through the testing lists
		for l in testing_lists:
			# build a table, excluding the list
			clf, labels = sci_train(os.path.join(os.getcwd(), 'training_emails'), save=False, exclusionary_list=l)
			# save the list to a file
			with open(os.path.join('bin', 'temp_cross_validation.txt'), 'w+') as test_file:
				for uid in l:
					test_file.write(os.path.relpath(uid) + '\n')
			# get the accuracy and output from testing the list
			accuracy, output = sci_test_accuracy(os.path.join('bin', 'temp_cross_validation.txt'), clf)
			total_accuracy += accuracy
			outputs += output
		# delete the temporary file
		os.remove(os.path.join('bin', 'temp_cross_validation.txt'))

		# average the accuracies
		total_accuracy /= num_tests
		actual = []
		predicted = []
		# print(labels)
		for line in outputs:
			actual.append(line[4])
			predicted.append(line[1])
			print('{}: {}, {}% {}(Correct: {}){}'.format(line[0], line[1], line[2], line[3], line[4], line[5]))
		if flags.make_visual:
			C = confusion_matrix(actual, predicted)
			# Plot normalized confusion matrix
			plot_confusion_matrix(C, total_accuracy*100, classes=np.unique(labels), normalize=True, title='Sci-Kit Bayes Normalized confusion matrix')
			if not os.path.isdir('graphs'):
				os.makedirs('graphs')
			plt.savefig('graphs/sci_confusion_matrix.png')
			# plt.show()
		print("Scikit Naive Bayes model {}-fold cross validation: {}%".format(num_tests, str(total_accuracy*100)))
		return True

def sci_train(training_path, save=True, exclusionary_list=[]):
	# train the model, akin to build_table function in classifier.py
	training = []
	labels = []
	# open the master dictonary
	master_dict = read_dictionary(os.path.join('bin', 'dictionary.txt'))
	words = master_dict.keys()
	for uid in get_first_uids(training_path):
		dict_file = os.path.join(uid, 'dictionary.txt')
		if uid not in exclusionary_list and os.path.isfile(os.path.join(uid, 'body_stripped.txt')) and os.path.isfile(dict_file):
			word_dict = read_dictionary(dict_file)
			training.append(build_vector(word_dict, words))
			catg_name = os.path.basename(os.path.abspath(os.path.join(uid, os.pardir, os.pardir)))
			labels.append(catg_name)
	# You can swap the model just by commenting/uncommenting the two lines below
	clf = MultinomialNB() # Bayes classifier
	# clf = MLPClassifier(max_iter=300, hidden_layer_sizes=(350,)) # multi-layer perceptron
	clf.fit(np.array(training), np.array(labels))
	if save:
		pickle.dump(clf, open(os.path.join('bin', 'clf.p'), 'wb'))
	return (clf, labels)

def sci_classify(word_dict, words, clf):
	# classify the word vector/dict, akin to the classify function in classifier.py
	vector = build_vector(word_dict, words)
	label = clf.predict(np.array(vector).reshape(1, -1))
	confidence = max(clf.predict_proba(np.array(vector).reshape(1, -1)).tolist()[0])
	return (label[0], confidence)

def sci_auto_classify(emails_path, test_path):
	# open the master dictonary
	master_dict = read_dictionary(os.path.join('bin', 'dictionary.txt'))
	# open the classifier
	if not os.path.isfile(os.path.join('bin', 'clf.p')):
		print('Error: no sci classifier found, run scib to build sci classifier')
		return
	clf_file = open(os.path.join('bin', 'clf.p'), 'rb')
	clf = pickle.load(clf_file)
	clf_file.close()
	unknown_path = os.path.join(os.getcwd(), 'training_emails', 'unknown')
	if not os.path.isdir(unknown_path):
		os.makedirs(unknown_path)
	if flags.redo:
		# move all emails from unknown category back into emails directory
		for ticket_id in os.listdir(unknown_path):
			os.rename(os.path.join(unknown_path, ticket_id), os.path.join(emails_path, ticket_id))
	extras = []
	for ticket_id in os.listdir(emails_path):
		if ticket_id == 'extra':
			continue
		# loop over the first uid of each ticket (for loop only runs once)
		for uid_path in get_first_uids(os.path.join(emails_path, ticket_id)):
			stripped_file = os.path.join(uid_path, 'body_stripped.txt')
			if not os.path.isfile(stripped_file):
				extras.append(ticket_id)
				break
			dict_file = os.path.join(uid_path, 'dictionary.txt')
			if not os.path.isfile(dict_file):
				break
			word_dict = read_dictionary(dict_file)
			# use the body_stripped file to categorize the email
			category, confidence = sci_classify(word_dict, master_dict.keys(), clf)
			# move the email to its respective category in test_emails
			if not category in os.listdir(test_path):
				os.makedirs(os.path.join(test_path, category))
			os.rename(os.path.join(emails_path, ticket_id), os.path.join(test_path, category, ticket_id))
	extra_file = open(os.path.join(os.getcwd(), os.path.join('categories', 'extra.txt')),'a+')
	if not os.path.isdir(os.path.join(emails_path, 'extra')):
		os.makedirs(os.path.join(emails_path, 'extra'))
	for ticket_id in extras:
		extra_file.write(ticket_id + '\n')
		os.rename(os.path.join(emails_path, ticket_id), os.path.join(emails_path, 'extra', ticket_id))
	extra_file.close()

def sci_test_accuracy(testing_uids, clf=None):
	correct = 0
	total = 0
	if not clf:
		clf_file = open(os.path.join('bin', 'clf.p'), 'rb')
		clf = pickle.load(clf_file)
		clf_file.close()
	# open the master dictionary
	master_dict = read_dictionary(os.path.join('bin', 'dictionary.txt'))
	words = master_dict.keys()
	test_data = []
	output = []
	with open(testing_uids, 'r') as testing_uids_file:
		# get list of uids to test from slist command
		testing_uids_list = testing_uids_file.read().split()
		for uid in toggle_tqdm(testing_uids_list):
			dict_file = os.path.join(uid, 'dictionary.txt')
			if not os.path.isfile(dict_file):
				continue
			word_dict = read_dictionary(dict_file)
			category = os.path.basename(os.path.abspath(os.path.join(uid, os.pardir, os.pardir)))
			guess, confidence = sci_classify(word_dict, words, clf)
			out = (uid, guess, str(confidence*100)[:4], '\33[41m', category, '\33[0m')
			if guess == category:
				out = (uid, guess, str(confidence*100)[:4], '\33[42m', category, '\33[0m')				
				correct += 1
			total += 1
			output.append(out)
	return correct/total, output

def build_vector(word_dict, words):
	vector = []
	for word in words:
		count = 0
		if word in word_dict:
			count = word_dict[word]
		vector.append(count)
	return vector