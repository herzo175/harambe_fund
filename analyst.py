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
DATA_KEYS = []
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

			self.load_datasets()
			#self.train()
			
			return


	def load_datasets(self):
		self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=self.MASTER_DATASET,
			target_dtype=np.int,
			features_dtype=np.float32
		)

		self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=self.TEST_DATASET,
			target_dtype=np.int,
			features_dtype=np.float32
		)

		print('training set:')
		print(self.training_set)
		print('test set:')
		print(self.test_set)


	def train(self):
		# Specify that all features have real-value data
		feature_columns = [tf.feature_column.numeric_column("x", shape=[self.DATA_WIDTH])]

		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
			hidden_units=[10, 20, 10],
			n_classes=self.N_CLASSES+1
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

	def test_against_current_data(self, symbol):
		stock_data = scrapers.data_to_list(
			scrapers.get_stock_data(symbol),
			self.DATA_KEYS
		)
		new_samples = np.array(
			[get_stock_data(symbol)], dtype=np.float32
		)
		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": new_samples},
			num_epochs=1,
			shuffle=False
		)

		predictions = list(self.classifier.predict(input_fn=predict_input_fn))
		predicted_classes = [p["classes"] for p in predictions]

		return predicted_classes


	def test(self, symbol):
		self.load_datasets()
		self.train()
		return self.test_against_current_data(symbol)


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
			
		return stock_data + [new_target]


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
				except:
					pass
		except:
			pass

		return data


	def create_earnings_dataset_range(self, datetime, days_ahead):
		data = []

		for i in range(0, days_ahead):
			date = datetime + timedelta(i)
			data.extend(self.create_earnings_dataset_date(date))
			
		return data