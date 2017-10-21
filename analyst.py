from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

class Analyst():
	def __init__(self):
		self.loadDataset()
		self.train()

	def loadDataset(self):
		if not os.path.exists(IRIS_TEST):
			raw = urllib.urlopen(IRIS_TRAINING_URL).read().decode('utf8')
			with open(IRIS_TRAINING, "w") as f:
				f.write(raw)

		if not os.path.exists(IRIS_TEST):
			raw = urllib.request.urlopen(IRIS_TEST_URL).read().decode('utf8')
			with open(IRIS_TEST, "w") as f:
				f.write(raw)

		self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=IRIS_TRAINING,
			target_dtype=np.int,
			features_dtype=np.float32
		)
	
		self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
			filename=IRIS_TEST,
			target_dtype=np.int,
			features_dtype=np.float32
		)

		print('training set:')
		print(self.training_set)
		print('test set:')
		print(self.test_set)

	def train(self):
		# Specify that all features have real-value data
		feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

		# Build 3 layer DNN with 10, 20, 10 units respectively.
		self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
			hidden_units=[10, 20, 10],
			n_classes=3,
			model_dir="/tmp/iris_model"
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
		accuracy_score = self.classifier.evaluate(input_fn=self.test_input_fn)["accuracy"]

		print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

	def test(self, testArr):
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

		print('accuracy:', self.classifier.evaluate(input_fn=predict_input_fn)["accuracy"])
		return predicted_classes
