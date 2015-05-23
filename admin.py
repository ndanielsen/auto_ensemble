#!/usr/local/bin/python
 # -*- coding: utf-8 -*-
"""
Data Cleaner and Controller for ensemble modeling

Author:
Nathan Danielsen
nathan.danielsen [at] gmail.com
"""

import pandas as pd

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler


class AutoEnsemble(object):

	def __init__(self, trainfile=None, testfile=None):

		self.trainfile = trainfile
		self.testfile = testfile


	@staticmethod
	def make_features(filename):
	    df = pd.read_csv(filename, index_col=0)
	    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
	    df['TitleLength'] = df.Title.apply(len)
	    df['BodyLength'] = df.BodyMarkdown.apply(len)
	    df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)

	    return df

	def create_files(self):

		self.train = self.make_features(self.trainfile)
		

		feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
		self.X = self.train[feature_cols]
		self.y = self.train.OpenStatus

	def scaler(self):
		scaler = StandardScaler()
		scaler.fit(self.X)
		self.X = scaler.transform(self.X)



	def randomtree(self):
		name = 'randomtree'
		rfclf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1)
		rfclf.fit(self.X_scaled, self.y)


		print rfclf.oob_score_


	def adabooster(self):
		name = 'adabooster'
		clf = AdaBoostClassifier(n_estimators=100)
		scores = cross_val_score(clf, self.X, self.y)

		print scores.mean()	


	def evaluate(self):
		"""
		Evaluates the accuracy and if the greatest of the log, then pickles the model.
		"""
		self.test =	make_features(self.testfile)
		pass




	def main(self):

		self.create_files()
		self.scaler()
		self.adabooster()

		# print self.y

		

		# print self.df.head(2)

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
	test = AutoEnsemble('fixtures/train.csv')
	print test.main()