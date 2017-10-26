from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from datetime import datetime, timedelta

import scrapers

MASTER_DATASET = "master.csv"
TEST_DATASET = "test.csv"
# Number of classifications; add 1 since lowest is 1
N_CLASSES = 4
DATA_KEYS = [
	'1y Target Est',
	'Beta',
	'EPS (TTM)',
	'PE Ratio (TTM)',
	'Buy Distance From Yearly Target'
]
DATA_WIDTH = len(DATA_KEYS)

class Analyst():
	def __init__(
		self,
		master_filename=MASTER_DATASET,
		test_filename=TEST_DATASET,
		num_classifications=N_CLASSES,
		data_keys=DATA_KEYS):
			self.MASTER_DATASET = master_filename
			self.TEST_DATASET = test_filename
			self.N_CLASSES = num_classifications
			self.DATA_KEYS = data_keys
			self.DATA_WIDTH = len(data_keys)

			self.training_set = {
				'set_data': [],
				'targets': []
			}

			self.test_set = {
				'set_data': [],
				'targets': []
			}

			self.load_datasets()
			self.train()

			return


	def load_datasets(
		self,
		master_filename=None,
		test_filename=None):

		master_filename = (
			self.MASTER_DATASET if master_filename is None else master_filename
		)
		test_filename = (
			self.TEST_DATASET if test_filename is None else test_filename
		)

		training_data = scrapers.CSV_to_2D_list(master_filename)
		test_data = scrapers.CSV_to_2D_list(test_filename)

		self.training_set['set_data'] = list(
			map(lambda r: list(map(lambda e: float(e), r[:-1])), training_data)
		)
		self.training_set['targets'] = list(
			map(lambda r: int(r[-1]), training_data)
		)

		self.test_set['set_data'] = list(
			map(lambda r: list(map(lambda e: float(e), r[:-1])), test_data)
		)
		self.test_set['targets'] = list(
			map(lambda r: int(r[-1]), test_data)
		)

		return self.training_set, self.test_set

	def save_datasets(
		self,
		training_set=None,
		test_set=None,
		master_filename=None,
		test_filename=None):

		training_set = (
			self.training_set if training_set is None else training_set
		)
		test_set = (
			self.test_set if test_set is None else test_set
		)
		master_filename = (
			self.MASTER_DATASET if master_filename is None else master_filename
		)
		test_filename = (
			self.TEST_DATASET if test_filename is None else test_filename
		)

		for i, r in enumerate(training_set['set_data']):
			row = self.create_earnings_dataset_row(
				r, training_set['targets'][i]
			)
			scrapers.add_to_csv(row, master_filename)

		for i, r in enumerate(test_set['set_data']):
			row = self.create_earnings_dataset_row(
				r, test_set['targets'][i]
			)
			scrapers.add_to_csv(row, test_filename)


	def train(self, training_set=None):
		training_set = (
			self.training_set if training_set is None else training_set
		)
		# Specify that all features have real-value data
		feature_columns = [
			tf.feature_column.numeric_column("x", shape=[self.DATA_WIDTH])
		]

		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.estimator.DNNClassifier(
			feature_columns=feature_columns,
			hidden_units=[20, 40, 20],
			n_classes=self.N_CLASSES+1
		)

		# Define the training inputs
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={'x': np.array(training_set['set_data'])},
			y=np.array(training_set['targets']),
			num_epochs=None,
			shuffle=True
		)

		# Train model.
		self.classifier.train(input_fn=train_input_fn, steps=2000)

		return self.classifier

	def accuracy(self, test_set=None):
		# Define the test inputs
		test_set = (
			self.test_set if test_set is None else test_set
		)

		def compare_results(row):
			result = self.test_against_current_data(row[:-1])

			return float(result[0][0]) >= float(row[-1])

		test_data = list(zip(test_set['set_data'], test_set['targets']))
		comparisons = list(
			map(
				lambda r: compare_results(r), test_data              
			)
		)

		successful_tests = list(filter(lambda e: e, comparisons))
		accuracy_score = len(successful_tests) / len(comparisons)

		print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
		return accuracy_score

	def test_against_current_data(self, stock_data):
		new_samples = np.array(
			[stock_data], dtype=np.float32
		)
		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": new_samples},
			num_epochs=1,
			shuffle=False
		)

		predictions = list(self.classifier.predict(input_fn=predict_input_fn))
		predicted_classes = [p['classes'] for p in predictions]

		return predicted_classes

	def test_symbol(self, symbol):
		stock_data = scrapers.data_to_list(
			scrapers.get_stock_data(symbol),
			self.DATA_KEYS
		)
		
		result = self.test_against_current_data(stock_data)

		return result[0][0]


	def test_symbol_raw(self, symbol):
		self.load_datasets()
		self.train()
		return self.test_symbol(symbol)


	def create_earnings_dataset_row(self, stock_data, target):
		# create row with stock data array and target
		# decide integer value for stock target
		if float(target) > 8:
			new_target = '4'
		elif float(target) > 0:
			new_target = '3'
		elif float(target) < 0 and float(target) > -8:
			new_target = '2'
		elif float(target) <= -8:
			new_target = '1'

		return list(map(lambda e: float(e), stock_data)) + [int(new_target)]


	def create_earnings_dataset_date(self, datetime):
		data = []

		try:
			earnings_rows = scrapers.get_earnings_calendar(datetime)

			for row in earnings_rows:
				try:
					stock_data = scrapers.data_to_list(
						scrapers.get_stock_data(row[0]),
						self.DATA_KEYS
					)
					dataset_row = self.create_earnings_dataset_row(
						stock_data, row[-1]
					)
					data.append(dataset_row)
				except Exception as e:
					print(e)
		except Exception as e:
			print(e)

		return data


	def create_earnings_dataset_range(
		self,
		datetime,
		days_ahead=0,
		dataset=None):
		#dataset should be either 'MASTER', 'TEST' or unspecified
		data = []

		for i in range(0, days_ahead):
			date = datetime + timedelta(i)
			data.extend(self.create_earnings_dataset_date(date))

		if dataset != None:
			set_data = list(map(lambda r: r[:-1], data))
			targets = list(map(lambda r: r[-1], data))

			if dataset == 'TEST':
				self.test_set['set_data'].extend(set_data)
				self.test_set['targets'].extend(targets)
			elif dataset == 'MASTER':
				self.training_set['set_data'].extend(set_data)
				self.training_set['targets'].extend(targets)

		return data
