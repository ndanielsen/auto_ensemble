#!/usr/local/bin/python
 # -*- coding: utf-8 -*-
"""
Data Cleaner and Controller for ensemble modeling

Author:
Nathan Danielsen
nathan.danielsen [at] gmail.com
"""

def import_data():

	pass


def clean_data():
	pass


"""
# add the predictions together
sum_of_preds = preds1 + preds2 + preds3 + preds4 + preds5

# ensemble predicts 1 (the "correct response") if at least 3 models predict 1
ensemble_preds = np.where(sum_of_preds >=3 , 1, 0)

# print the ensemble's first 20 predictions
print ensemble_preds[:20]

"""