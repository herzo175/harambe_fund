from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf
from yahoo_finance import Share

MASTER_DATASET = "master.csv"
TEST_DATASET = "test.csv"

class Analyst():
	def __init__(self):
		# self.load_datasets()
		# self.train()
		return

	def load_datasets(self):
		self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=MASTER_DATASET,
			target_dtype=np.float32,
			features_dtype=np.float32
		)

		self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=TEST_DATASET,
			target_dtype=np.float32,
			features_dtype=np.float32
		)

		print('training set:')
		print(self.training_set)
		print('test set:')
		print(self.test_set)

	def add_to_dataset(self, symbol, target, type='MASTER'):
		data = []
		stock = Share(symbol)

		data.append(stock.get_price() or '0')
		data.append(stock.get_avg_daily_volume() or '0')
		data.append(stock.get_market_cap()[:-1] or '0')
		data.append(stock.get_earnings_share() or '0')
		data.append(stock.get_year_high() or '0')
		data.append(stock.get_year_low() or '0')
		data.append(stock.get_change_from_50_day_moving_average() or '0')
		data.append(stock.get_dividend_share() or '0')
		data.append(stock.get_price_earnings_ratio() or '0')
		data.append(stock.get_price_earnings_growth_ratio() or '0')
		data.append(stock.get_EPS_estimate_current_year() or '0')
		data.append(stock.get_EPS_estimate_next_quarter() or '0')
		data.append(stock.get_EPS_estimate_next_year() or '0')
		data.append(stock.get_one_yr_target_price() or '0')

		data.append(str(target))

		new_string =  ','.join(data) + '\n'

		if len(new_string.split(',')) == 15:
			f = open(TEST_DATASET if type == 'TEST' else MASTER_DATASET, "a")
			f.write(new_string)

		return new_string

	def train(self):
		# Specify that all features have real-value data
		feature_columns = [tf.feature_column.numeric_column("x", shape=[14])]

		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
			hidden_units=[10, 20, 10],
			n_classes=3,
			model_dir="/tmp/stock_model"
		)

		# Define the training inputs
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": np.array(self.training_set.data)},
			y=np.array(self.training_set.target),
			num_epochs=None,
			shuffle=True
		)

		# Train model.
		self.classifier.train(input_fn=train_input_fn, steps=2000)

		# Define the test inputs
		test_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": np.array(self.test_set.data)},
			y=np.array(self.test_set.target),
			num_epochs=1,
			shuffle=False
		)

		# Evaluate accuracy.
		accuracy_score = self.classifier.evaluate(input_fn=test_input_fn)["accuracy"]

		print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

	def test_against_current_data(self, testArr):
		new_samples = np.array(
			[testArr], dtype=np.float32
		)
		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": new_samples},
			num_epochs=1,
			shuffle=False
		)

		predictions = list(self.classifier.predict(input_fn=predict_input_fn))
		predicted_classes = [p["classes"] for p in predictions]

		return predicted_classes

	def test(self, testArr):
		self.load_datasets()
		self.train()
		return self.test_against_current_data(testArr)


def reticulate_splines(analyst):
	# add to master data (ticker, surprise)
	data = [
		('ETFC', -31.37), ('LION', -23.08), ('LHO', -22.86),
		('NAP', -14.29), ('IBKC', -13.79), ('HBMD', -13.04),
		('CZNC', -5.88), ('BBT', -5.13), ('BANF', -4.23),
		('WERN', -3.13), ('ABCB', -3.08), ('INDB', -2.25),
		('POOL', .87), ('SXT', 1.14), ('WAL', 1.28),
		('NUE', 1.28), ('SASR', 1.59), ('BK', 2.17),
		('ENFC', 2.7), ('SON', 2.7), ('DGX', 2.96),
		('NCR', 3.33), ('EWBC', 3.49), ('DOV', 3.57),
		('HA', 3.78), ('SHBI', 3.85), ('PBCT', 4),
		('WDFC', 4.12), ('SBNY', 4.57), ('WBS', 4.69),
		('TXT', 4.84), ('NVR', 4.85), ('WFBI', 4.88),
		('FFBC', 5.26), ('DHR', 5.26), ('AMNB', 5.77),
		('ADS', 6.15), ('PYPL', 6.98), ('CHMG', 7.04),
		('MXIM', 7.14), ('MBFI', 7.69), ('MRTN', 7.69),
		('BMTC', 8.33), ('STBA', 8.33), ('BCBP', 8.7),
		('LOAN', 9.09), ('WBC', 11.76), ('ATHN', 12),
		('COBZ', 12.5), ('WGO', 12.86), ('ASB', 13.89),
		('MSA', 15), ('TSC', 16.67), ('CCNE', 20.51),
		('BX', 27.78), ('CAI', 28.57), ('GATX', 28.87),
		('SKX', 37.21), ('PFPT', 38.89), ('ISRG', 39.2),
		('TRV', 102.22), ('CLW', 190.91)
	]

	test = [
		('RNST', -13.11), ('SNV', -10.77), ('SCSS', -8.82),
		('HBHC', -8.11), ('PNFP', -7.78), ('IBM', 0.61),
		('HOG', 2.56), ('GFED', 2.63), ('OMC', 2.73),
		('UBNK', 3.45), ('FULT', 3.7), ('UNH', 3.91),
		('CMA', 5), ('UFPI', 5.13), ('LRCX', 5.49),
		('JNJ', 5.56), ('MBWM', 6.25), ('NAVI', 10),
		('PACW', 12), ('NCBS', 12.35), ('IBKR', 13.16),
		('GWW', 13.28), ('PZN', 13.33), ('MS', 14.81),
		('BCLI', 18.75), ('GS', 20.38), ('PGR', 20.59),
		('ADTN', 26.92), ('SYNT', 38.1)
	]

	# clear files
	f = open(MASTER_DATASET, 'w')
	f.write(
		str(len(data)) + ',14\n'
	)
	f = open(TEST_DATASET, 'w')
	f.write(
		str(len(test)) + ',14\n'
	)

	for t in data:
		print(t)
		analyst.add_to_dataset(t[0], t[1])

	for t in test:
		print(t)
		analyst.add_to_dataset(t[0], t[1], 'TEST')
