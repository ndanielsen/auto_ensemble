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
from sklearn.neighbors import KNeighborsClassifier
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
		trainfile = 'fixtures/train_features.pickle'
		
		if os.path.isfile(trainfile):
			self.df = pd.read_pickle(trainfile)
			self.y = self.df.OpenStatus
		else:
			self.df = self.make_features(self.trainfile)
			self.openstatus_count = pd.DataFrame({'OpenStatus_Count' : self.df.groupby([u'OwnerUserId']).OpenStatus.sum()  }).reset_index()
			self.df = pd.merge(left=self.df, right=self.openstatus_count, how='left', left_on='OwnerUserId', right_on='OwnerUserId')			
			self.df['OpenStatus_Count'] = self.df['OpenStatus_Count'].fillna(0)
			self.y = self.df.OpenStatus
			self.df.to_pickle(trainfile)

		
		testfile =  'fixtures/tested_features.pickle'
		if os.path.isfile(testfile):
			self.X_test = pd.read_pickle(testfile)
		else:
			self.X_test = self.make_features(self.testfile)
			self.openstatus_count = pd.DataFrame({'OpenStatus_Count' : self.df.groupby([u'OwnerUserId']).OpenStatus.sum()  }).reset_index()
			self.X_test = pd.merge(left=self.X_test, right=self.openstatus_count, how='left', left_on='OwnerUserId', right_on='OwnerUserId')
			self.X_test['OpenStatus_Count'] = self.X_test['OpenStatus_Count'].fillna(0)
			self.X_test.to_pickle(testfile)

		self.y_pred = [0 for elem in range(0, self.X_test.shape[0])]

	def make_features(self, filename):
		df = pd.read_csv(filename)
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

		def clean_text(x):
			text = bleach.clean(x, strip=True)
			return unicode(text)
		df['Cleaned_BodyMarkdown'] = df.BodyMarkdown.apply(clean_text)

		# def first(x):
		# 	sent = nltk.sent_tokenize(x)
		# 	return sent[0]
		# df['First_Sentence'] = df.Cleaned_BodyMarkdown.apply(first)

		# def last(x):
		# 	sent = nltk.sent_tokenize(x)
		# 	return sent[-1]
		# df['Last_Sentence'] = df.Cleaned_BodyMarkdown.apply(last)




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

		sub = pd.DataFrame({'id':self.X_test.PostId, 'OpenStatus':probas}).set_index('id')
		sub.to_csv('sub.csv')

	def pickle_loader(self, pickle_file=None):
		with open(pickle_file, "r") as fp: 	#Load model from file
			grid = pickle.load(fp)
			return grid

	def pickler(self, model=None, name=None, feature_cols=None):
		
		pickle_file = 'models/' + name + ''.join(feature_cols) + '.pickle'
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

		self.pickler(model=grid, name=cname, feature_cols=feature_cols)

		self.y_pred += grid.predict_proba(self.X_test[feature_cols])[:, 1]
		self.count_models += 1

		print cname, feature_cols, grid.best_score_

	def text_logisticregression_load(self, feature_cols=None):
		cname = 'text_logisticregression'
		pickle_file = 'models/' + cname + ''.join(feature_cols) + '.pickle'

		if os.path.isfile(pickle_file):
			grid = self.pickle_loader(pickle_file=pickle_file)
			print 'pickle loaded'
		else:

			print 'pickle not loaded'
			self.X = self.df[feature_cols]
			
			pipe = make_pipeline(CountVectorizer(ngram_range=(1, 1), stop_words='english'), 
				LogisticRegression())

			c_range = arange(.1, 1, .1)
			p_options = ['l2', 'l1']
			param_grid = dict(logisticregression__C=c_range, logisticregression__penalty=p_options)
			grid = GridSearchCV(pipe, param_grid, cv=5, scoring='log_loss')
			grid.fit(self.X, self.y)		

			self.logger(name=cname, score=grid.best_score_, best=grid.best_params_, tested_feature_cols=feature_cols)

			self.pickler(model=grid, name=cname, feature_cols=feature_cols)

		self.y_pred += grid.predict_proba(self.X_test[feature_cols])[:, 1]
		self.count_models += 1

		print cname, feature_cols


	def knn(self, feature_cols=None):
		cname = 'knn'

		pickle_file = 'models/' + cname + ''.join(feature_cols) + '.pickle'

		self.X = self.df[feature_cols]
		self.y_test = self.X_test[feature_cols]
		scaler = StandardScaler()
	
		scaler.fit(self.X)
		self.X = scaler.transform(self.X)
		self.y_test = scaler.transform(self.y_test)


		if os.path.isfile(pickle_file):
			grid = self.pickle_loader(pickle_file=pickle_file)
			print 'pickle loaded'

		else:
			print 'pickle not loaded'

			neighbors_range = range(1, 100, 1)
			weight_options = ['uniform', 'distance']
			param_grid = dict(n_neighbors=neighbors_range, weights=weight_options)

			knn = KNeighborsClassifier()

			grid = GridSearchCV(knn, param_grid, cv=5, scoring='log_loss')
			grid.fit(self.X, self.y)
			

			self.logger(name=cname, score=grid.best_score_, best=grid.best_params_, tested_feature_cols=feature_cols)

			self.pickler(model=grid, name=cname, feature_cols=feature_cols)




		self.y_pred += grid.predict_proba(self.y_test)[:, 1]
		self.count_models += 1

		print cname, feature_cols


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

		# self.pickler(model=grid, name=cname, feature_cols='NumericData')
		
		self.y_pred += grid.predict_proba(self.y)[:, 1]
		self.count_models += 1

	def randomforest_XXX(self):
		feature_cols = ['ReputationAtPostCreation', 'OpenStatus_Count', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
		self.X = self.df[feature_cols]


		# grid = self.pickle_loader(pickle_file='models/randomforestNumericData53_.pickle')

		RandomForestClassifier()

		scaler = StandardScaler()
		scaler.fit(self.X)
		self.y = scaler.transform(self.X_test[feature_cols])


		self.y_pred += grid.predict_proba(self.y)[:, 1]
		self.count_models += 1

	def randomforest_53(self):
		feature_cols = ['ReputationAtPostCreation', 'OpenStatus_Count', 'Answers', 'TitleLength', 'BodyLength', 'NumTags']
		self.X = self.df[feature_cols]

		rfclf = RandomForestClassifier(n_estimators=240, max_features=3, oob_score=True, random_state=1)
		
		rfclf.fit(self.X, self.y)

		self.y_pred += rfclf.predict_proba(self.X_test[feature_cols])[:, 1]
		self.count_models += 1

		print 'done'





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
		self.text_logisticregression_load(feature_cols='Title')
		# self.text_logisticregression(feature_cols='Cleaned_BodyMarkdown')
		# self.text_logisticregression(feature_cols='First_Sentence')
		# self.text_logisticregression(feature_cols='Last_Sentence')

		self.knn(feature_cols=['Answers', 'ReputationAtPostCreation'])

		self.submission()

		pass





if __name__ == '__main__':
	print 'hello'
	test = AutoEnsemble(trainfile='fixtures/train.csv', testfile='fixtures/test.csv', message='Tag1')
	print test.main()
	# print test.df.columns
	# print test.df['Title']