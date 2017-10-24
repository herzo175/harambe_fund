from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf
from scrapers import get_stock_data
from yahoo_finance import Share

MASTER_DATASET = "master.csv"
TEST_DATASET = "test.csv"
DATA_WIDTH = 12
# Number of classifications; add 1 since lowest is 1
N_CLASSES = 4

class Analyst():
	def __init__(self):
		self.load_datasets()
		self.train()
		return


	def load_datasets(self):
		self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=MASTER_DATASET,
			target_dtype=np.int,
			features_dtype=np.float32
		)

		self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=TEST_DATASET,
			target_dtype=np.int,
			features_dtype=np.float32
		)

		print('training set:')
		print(self.training_set)
		print('test set:')
		print(self.test_set)


	def train(self):
		# Specify that all features have real-value data
		feature_columns = [tf.feature_column.numeric_column("x", shape=[DATA_WIDTH])]

		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
			hidden_units=[10, 20, 10],
			n_classes=N_CLASSES+1
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