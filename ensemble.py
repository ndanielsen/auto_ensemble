#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import csv
import datetime 
import time
import os
import pickle

import bleach
import nltk
from textblob import TextBlob

import numpy as np

from numpy import arange

import pandas as pd

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



class AutoEnsemble(object):

	def __init__(self, trainfile=None, testfile=None, message=None):

		self.trainfile = trainfile
		self.testfile = testfile
		self.message = message
		self.timestamp = datetime.datetime.fromtimestamp(time.time())
		self.feature_files()
		self.count_models = 0

		


	def feature_files(self):
		trainfile = 'fixtures/train_features.csv'
		
		if os.path.isfile(trainfile):
			self.df = pd.read_csv(trainfile, index_col=0)
			self.dfnum = self.df._get_numeric_data()
			self.y = self.df.OpenStatus
		else:
			self.df = self.make_features(self.trainfile)
			self.openstatus_count = pd.DataFrame({'OpenStatus_Count' : self.df.groupby([u'OwnerUserId']).OpenStatus.sum()  }).reset_index()
			self.df = pd.merge(self.df, self.openstatus_count, on=['OwnerUserId'])			

			self.y = self.df.OpenStatus
			self.df.to_csv(trainfile)

		
		testfile =  'fixtures/tested_features.csv'
		if os.path.isfile(testfile):
			self.X_test = pd.read_csv(testfile, index_col=0)
			self.X_testnum = self.X_test._get_numeric_data()
		else:
			self.X_test = self.make_features(self.testfile)
			self.openstatus_count = pd.DataFrame({'OpenStatus_Count' : self.df.groupby([u'OwnerUserId']).OpenStatus.sum()  }).reset_index()
			self.X_test = pd.merge(self.X_test, self.openstatus_count, on=['OwnerUserId'])
			self.X_test.to_csv(testfile)

		self.y_pred = [0 for elem in range(0, self.X_test.shape[0])]

	def make_features(self, filename):
		df = pd.read_csv(filename, index_col=0)
		df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
		df['TitleLength'] = df.Title.apply(len)
		df['BodyLength'] = df.BodyMarkdown.apply(len)
		df['NumTags'] = df.loc[:, 'Tag1':'Tag5'].notnull().sum(axis=1)
		df['Tag1'] = df.Tag1.fillna('None')
		df['BodySentences_num'] = df.BodyMarkdown.apply(lambda x: len(TextBlob(x.decode('utf-8')).sentences))
		datetime_cols = ['OwnerCreationDate', 'PostCreationDate'] #, 'PostClosedDate']
		for col in datetime_cols:
			df[col] = pd.to_datetime(df[col])
			df[col + '_Year'] = pd.DatetimeIndex(df[col]).year
			df[col + '_Month'] = pd.DatetimeIndex(df[col]).month
			df[col + '_Day'] = pd.DatetimeIndex(df[col]).day
			df[col + '_DayofWeek'] = pd.DatetimeIndex(df[col]).dayofweek
			df[col + '_Hour'] = pd.DatetimeIndex(df[col]).hour

		#df['CreatetoPostDateDelta'] = df['PostCreationDate'] - df['OwnerCreationDate']
		#df['CreatetoPostDateDelta_seconds'] = df['CreatetoPostDateDelta'] / np.timedelta64(1, 's')
		# openstatus_count = pd.DataFrame({'OpenStatus_Count' : df.groupby([u'OwnerUserId']).OpenStatus.sum()  }).reset_index()
		# df = pd.merge(df, openstatus_count, on=['OwnerUserId'])
		def first(x):
			text = bleach.clean(x, strip=True)
			return text
		df['Cleaned_BodyMarkdown'] = df.BodyMarkdown.apply(first)



		return df

	def logger(self, name=None, score=None, best=None, tested_feature_cols=None):
		"""
		Checks the log of modeling and adds an entry. 
		If the scoring is the highest on file, it creates a pickle file of the classifer.

		"""
		filename = 'models/score_log.txt'
		data = [name, score, str(self.timestamp), self.message, best, tested_feature_cols]
		with open(filename, 'a+') as f:
			csv_writer = csv.writer(f)
			csv_writer.writerow(data)

	def submission(self):
		"""
		Sums and averages all predicted probablities and exports to submission CSV.
		"""

		probas = self.y_pred / self.count_models

		sub = pd.DataFrame({'id':self.X_test.index, 'OpenStatus':probas}).set_index('id')
		sub.to_csv('sub.csv')

	def pickle_loader(self, pick_file=None):
		if os.path.isfile(pickle_file):
			with open(pickle_file, "r") as fp: 	#Load model from file
				grid.best_estimator_ = pickle.load(fp)
				return grid.best_estimator_

	def pickler(self, model=None, name=None, feature_cols=None):
		pickle_file = 'models/' + name + feature_cols + '_.pickle'
		with open(pickle_file, "w+") as fp:
			pickle.dump(model.best_estimator_, fp)

	def text_logisticregression(self, feature_cols=None, picklename=None):

		cname = 'text_logisticregression'
		self.X = self.df[feature_cols]
		
		pipe = make_pipeline(CountVectorizer(ngram_range=(1, 1), stop_words='english'), 
			LogisticRegression())

		c_range = arange(.1, 1, .1)
		p_options = ['l2', 'l1']
		param_grid = dict(logisticregression__C=c_range, logisticregression__penalty=p_options)
		grid = GridSearchCV(pipe, param_grid, cv=5, scoring='log_loss')
		grid.fit(self.X, self.y)		

		self.logger(name=cname, score=grid.best_score_, best=grid.best_params_, tested_feature_cols=feature_cols)


		self.pickler(model=grid, name=cname, feature_cols='BodyMarkdown')

		return grid.best_estimator_

	def randomforest(self, feature_cols=None):
		cname = 'randomforest'
		self.X = self.df[feature_cols]

		max_range = range(3, 5)
		n_ests = range(175, 250, 5)
		param_grid = dict(randomforestclassifier__max_features=max_range, randomforestclassifier__n_estimators=n_ests)

		pipe = make_pipeline(
			StandardScaler(), 
			RandomForestClassifier())
					
		grid = GridSearchCV(pipe, param_grid, cv=3, scoring='log_loss')
		print 'ok'
		grid.fit(self.X, self.y)		

		self.logger(name=cname, score=grid.best_score_, best=grid.best_params_, tested_feature_cols=feature_cols)

		self.pickler(model=grid, name=cname, feature_cols='NumericData')
		
	def randomforest_53():
		feature_cols = ['ReputationAtPostCreation', 'OpenStatus_Count', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']


		grid = pickle_loader(pickle_file='53randomforestNumericData_.pickle')

		self.y_pred += grid.predict_proba(self.X_test[feature_cols])[:, 1]
		self.count_models += 1

	def ensemble(self):
		"""
		INIT -- For each proposed model, checks to see if a unique pickle has created for it.

		Branch True : If proposed model has pickle, then it loads the pickle and makes a prediction that feeds into the ensemble.

		Branch False : If there is not pickle for proposed model, then model is fitted with test data and pickle is saved.

		"""
		self.randomforest_53()


		# if pickled:
		# 	pickle_loader(name=None, feature_cols='')				
		# 	self.y_pred += grid.predict_proba(self.X_test[feature_cols])[:, 1]
		# 	self.count_models += 1
 
		# else:
		# 	grid = model
		# 	self.y_pred += grid.predict_proba(self.X_test[feature_cols])[:, 1]
		# 	self.count_models += 1


	def main(self):
		# Feature Columns to test ['Title', 'Tag1', 'FirstFourSentences']
		# self.text_logisticregression(feature_cols='Tag1')
		# self.text_logisticregression(feature_cols='Title')
		# self.text_logisticregression(feature_cols='BodyMarkdown')

		# self.randomforest(feature_cols=['ReputationAtPostCreation', 'OpenStatus_Count', 'Answers', 'TitleLength', 'BodyLength', 'NumTags'])

		self.submission()

		pass





if __name__ == '__main__':
	print 'hello'
	test = AutoEnsemble(trainfile='fixtures/train.csv', testfile='fixtures/test.csv', message='Tag1')
	print test.main()
	# print test.df.columns
	# print test.df['Title']