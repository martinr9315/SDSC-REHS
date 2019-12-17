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
import csv

class Table:
    def __init__(self):
        self.columns = []
        self.titles = []
        self.classifiers = []
        self.row_labels = []

    def add_row(self, row, classifier, label):
        self.columns.append(row)
        self.classifiers.append(classifier)
        self.row_labels.append(label)

    def add_column(self, title):
        for row in self.columns:
            row.append(0)
        self.titles.append(title)

    def get_classifier_indices(self, classifier):
        return [i for i, x in enumerate(self.classifiers) if x == classifier]

    def get_from_pos(self, x, y):
        return self.columns[y][x]

    def get_row(self, index):
        return self.columns[index]

    def get_titles(self):
        return self.titles

    def get_classifiers(self):
        return self.classifiers

    def get_table(self):
        return self.columns

    def get_row_labels(self):
        return self.row_labels

    def get_from_data(self, title, row_label):
        return self.columns[self.row_labels.index(row_label)][self.titles.index(title)]

    def set_from_pos(self, x, y, value):
        self.columns[y][x] = value

    def set_from_data(self, title, row_label, value):
        self.columns[self.row_labels.index(row_label)][self.titles.index(title)] = value
    import csv

    def save_csv(self, path, classifier_map={}):
        with open(path, 'w+') as output:
            wr = csv.writer(output, quoting=csv.QUOTE_ALL)
            wr.writerow(['classifier'] + self.titles)
            for i in range(len(self.columns)):
                row = self.columns[i]
                classifier = self.classifiers[i]
                if classifier_map:
                    wr.writerow([classifier_map.index(classifier)] + row)
                else:
                    wr.writerow([classifier] + row)