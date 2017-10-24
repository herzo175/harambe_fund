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

"""
class ExampleAnalyst():
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
		accuracy_score = self.classifier.evaluate(input_fn=test_input_fn)["accuracy"]

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

		return predicted_classes
"""

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()