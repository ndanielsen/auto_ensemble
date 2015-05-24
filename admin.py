#!/usr/local/bin/python
 # -*- coding: utf-8 -*-
"""
Data Cleaner and Controller for ensemble modeling

Author:
Nathan Danielsen
nathan.danielsen [at] gmail.com
"""
import csv
import datetime 
import time

from numpy import arange

import pandas as pd

from sklearn import metrics

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import StandardScaler



class AutoEnsemble(object):

	def __init__(self, trainfile=None, testfile=None, message=None):

		self.trainfile = trainfile
		self.testfile = testfile
		self.message = message

		self.timestamp = datetime.datetime.fromtimestamp(time.time())

	@staticmethod
	def make_features(filename):
	    df = pd.read_csv(filename, index_col=0)
	    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
	    df['TitleLength'] = df.Title.apply(len)
	    df['BodyLength'] = df.BodyMarkdown.apply(len)
	    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
	    return df

	def logger(self, name=None, score=None, best=None):
		"""
		Checks the log of modeling and adds an entry. 
		If the scoring is the highest on file, it creates a pickle file of the classifer.

		"""
		filename = 'models/score_log.txt'

		data = [name, score, str(self.timestamp), self.message, best]
		with open(filename, 'a+') as f:
			csv_writer = csv.writer(f)
			csv_writer.writerow(data)

	def create_files(self):

		self.train = self.make_features(self.trainfile)
		feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
		self.X = self.train[feature_cols]
		self.y = self.train.OpenStatus

	def naivebayes(self):
		cname = 'naivebayes'

		self.train = self.make_features(self.trainfile)
		feature_cols = ['Title']
		self.X = self.train.Title
		self.y = self.train.OpenStatus

		vect = CountVectorizer(ngram_range=(1, 1), stop_words='english')
		
		counts = vect.fit_transform(self.X)

		nb = MultinomialNB()
		
		scores = cross_val_score(nb, counts, self.y, cv=3, scoring='log_loss')

		self.logger(name=cname, score=str(scores.mean()))

	def logisticregression(self):

		cname = 'logisticregression'

		self.train = self.make_features(self.trainfile)
		feature_cols = ['Title']
		self.X = self.train.BodyMarkdown
		self.y = self.train.OpenStatus

		vect = CountVectorizer(ngram_range=(1, 1), stop_words='english')
		
		counts = vect.fit_transform(self.X)

		# logreg = LogisticRegression(C=0.1, penalty='l1')

		# scores = cross_val_score(logreg, counts, self.y, cv=3, scoring='log_loss')

		# self.logger(name=cname, score=str(scores.mean()))


		logreg = LogisticRegression()

		c_range = arange(.1, 1, .1)
		p_options = ['l1', 'l2']
		param_grid = dict(C=c_range, penalty=p_options)
		grid = GridSearchCV(logreg, param_grid, cv=3, scoring='log_loss')
		grid.fit(counts, self.y)		

		self.logger(name=cname, score=grid.best_score_, best=grid.best_params_)



	def scaler(self):
		scaler = StandardScaler()
		scaler.fit(self.X)
		self.X = scaler.transform(self.X)

	def knn(self):
		cname = 'knn'
		neighbors_range = range(100, 200, 5)
		weight_options = ['uniform']#, 'distance']
		param_grid = dict(n_neighbors=neighbors_range, weights=weight_options)

		knn = KNeighborsClassifier()

		grid = GridSearchCV(knn, param_grid, cv=5, scoring='log_loss')
		grid.fit(self.X, self.y)
		self.logger(name=cname, score=grid.best_score_, best=grid.best_params_)


	def randomforest(self):
		cname = 'randomforest'
		depth_range = range(1, 5)
		leaf_range = range(1, 11)
		param_grid = dict(max_depth=depth_range, min_samples_leaf=leaf_range)
		
		rfclf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
		grid = GridSearchCV(rfclf, param_grid, cv=5, scoring='log_loss')
		grid.fit(self.X, self.y)

		self.logger(name=cname, score=grid.best_score_, best=grid.best_params_)


	def adabooster(self):
		cname = 'adabooster'
		clf = AdaBoostClassifier(n_estimators=100)
		scores = cross_val_score(clf, self.X, self.y, scoring='log_loss')

		self.logger(name=cname, score=str(scores.mean()))	


	def gradientbooster(self):
		cname = 'gradientbooster'
		gbclf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0)
		scores = cross_val_score(gbclf, self.X, self.y, scoring='log_loss')

		self.logger(name=cname, score=str(scores.mean()))


	def evaluate(self):
		"""
		Evaluates the accuracy and if the greatest of the log, then pickles the model.
		"""
		self.test =	make_features(self.testfile)
		
		y_prob = grid.predict_proba(test[feature_cols])[:, 1]
		sub = pd.DataFrame({'id':test.index, 'OpenStatus':y_prob}).set_index('id')
		sub.to_csv('sub.csv')

		pass




	def main(self):

		self.create_files()
		# self.scaler()
		# self.naivebayes()
		self.logisticregression()
		# self.knn()
		# self.randomforest()
		# self.adabooster()
		# self.gradientbooster()


"""
# add the predictions together
sum_of_preds = preds1 + preds2 + preds3 + preds4 + preds5

# ensemble predicts 1 (the "correct response") if at least 3 models predict 1
ensemble_preds = np.where(sum_of_preds >=3 , 1, 0)

# print the ensemble's first 20 predictions
print ensemble_preds[:20]

# how accurate was the ensemble?
ensemble_preds.mean()

"""

### Numeric data only



### pass to modelings


### Evaluation esemble modeling


### Log loss

### Write evluation to file

### If evaluation is best, create submission file with timestamp


if __name__ == '__main__':
	print 'hello'
	test = AutoEnsemble('fixtures/train.csv', message='No max depth')
	print test.main()