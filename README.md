#Auto Ensemble : Because it's easier?

_Why not build a class based framework for ensembling instead of doing my GA class project?_

__Make some that you have the kaggle data files download into fixtures__



## Intended -- Eventual Use:


import AutoEnsemble

test = AutoEnsemble(trainfile='fixtures/train.csv', testfile='fixtures/test.csv', message="Testing something" )

test.logisticalregression(feature_cols=[some features])

test.randomforest(feature_cols=[some features])

test.knn(feature_cols=[some features])

test.submission() #produces an submissions file with unweighted probabilities.