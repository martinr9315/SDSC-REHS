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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import json
import ast
import pickle
from table import Table

from dataset import get_first_uids
from classifier import build_table

def build_csv(training_path, testing_path, training_table):
    with open(os.path.join('bin', 'testing_list.txt'), 'r') as test_emails:
        testing_list = []
        for email in test_emails.read().split():
            testing_list.append(email)
        training_table.save_csv(training_path)
        training_emails = []
        for email in get_first_uids(os.path.join(os.getcwd(), 'training_emails')):
            if os.path.relpath(email) not in testing_list:
                training_emails.append(email)
        testing_table = build_table(os.getcwd(), save=False, exclusionary_list=training_emails)
        testing_table.save_csv(testing_path)

def build_pie_chart(category, csv_path):
    """
    builds pie charts displaying the top 9 most frequent words of the emails of each category
    """
    with open(csv_path) as csv_file:
        vectors = csv_file.read().split()
        vectors = [v.split(',') for v in vectors]
        temp = []
        for v in vectors:
            temp.append([v2.strip('"') for v2 in v])
        labels = temp[0][1:]
        vectors = [t[1:] for t in temp[1:] if t[0] == category]
        if len(vectors) == 0:
            return
        sizes = []
        for i in range(len(vectors[0])):
            s = 0
            for v in vectors:
                s += int(v[i])
            sizes.append(s)
        label_sizes = []
        for i in range(len(sizes)):
            label_sizes.append((labels[i], sizes[i]))
        label_sizes.sort(reverse=True, key=lambda x: x[1])
        labels = [ls[0] for ls in label_sizes]
        sizes = [ls[1] for ls in label_sizes]
        other_size = sum(sizes[10:])
        sizes = sizes[:9]
        sizes.append(other_size)
        labels = labels[:9]
        labels.append('others')
        explode = []
        for i in range(9):
            explode.append(0)
        explode.append(0.1)
        plt.cla()
        plt.axis('equal')
        plt.title('Frequent words in ' + category.replace('_', ' '))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode)
        if not os.path.isdir('graphs'):
            os.makedirs('graphs')
        plt.savefig('graphs/' + category + '_pie.png')

def plot_confusion_matrix(cm, accuracy, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('accuracy', rotation=270, labelpad=20)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = 0.7
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if(cm[i, j] >  0.1):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center", size=5,
                         color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}%'.format(accuracy))