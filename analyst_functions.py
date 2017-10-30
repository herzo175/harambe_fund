from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from datetime import datetime, timedelta

import utils

"""
  DEFAULT DATA
"""

# Number of classifications; add 1 since lowest is 1
N_CLASSES = 4
def CLASSIFICATION_FUNCTION(target):
  if float(target) > 8:
    new_target = '4'
  elif float(target) > 0:
    new_target = '3'
  elif float(target) < 0 and float(target) > -8:
    new_target = '2'
  elif float(target) <= -8:
    new_target = '1'

  return new_target

DATA_KEYS = [
  '1y Target Est',
  'Beta',
  'EPS (TTM)',
  'PE Ratio (TTM)',
  'Buy Distance From Yearly Target'
]
DATA_WIDTH = len(DATA_KEYS)

"""
  PUBLIC METHODS
"""

def train(training_set, data_width, n_classes):
  # data_width is length of input array when classifying raw data
  # n_classes are natural numbers from 1, (1, 2, 3, 4 = 4 classes)
  # Specify that all features have real-value data
  feature_columns = [
    tf.feature_column.numeric_column("x", shape=[data_width])
  ]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[20, 40, 20],
    n_classes=n_classes+1
  )

  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array(training_set['set_data'])},
    y=np.array(training_set['targets']),
    num_epochs=None,
    shuffle=True
  )

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  return classifier

def accuracy(test_set, classifier):
  def compare_results(row):
    result = test_data(row[:-1], classifier)

    return float(result[0][0]) >= float(row[-1])

  test_results = list(zip(test_set['set_data'], test_set['targets']))
  comparisons = list(
    map(
      lambda r: compare_results(r), test_results            
    )
  )

  successful_tests = list(filter(lambda e: e, comparisons))
  accuracy_score = len(successful_tests) / len(comparisons)

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
  return accuracy_score


def test_data(stock_data, classifier):
  new_samples = np.array(
    [stock_data], dtype=np.float32
  )
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False
  )

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p['classes'] for p in predictions]

  return predicted_classes


def test_symbol(symbol, data_keys, classifier):
  stock_data = utils.get_stock_data(symbol)
  list_stock_data = utils.data_to_list(
    stock_data,
    data_keys
  )
    
  result = test_data(list_stock_data, classifier)

  return int(result[0][0])


def create_earnings_dataset_row(
  stock_data,
  target,
  classification_function):
  # create row with stock data array and target
  # optional parameter for custom classification function
  new_target = classification_function(target)

  return list(map(lambda e: float(e), stock_data)) + [int(new_target)]


def create_earnings_dataset_date(
  datetime,
  data_keys,
  classification_function):
  # get current stock data and results for single date
  data = []

  try:
    earnings_rows = list(
      filter(
        lambda r: '-' not in r, utils.get_earnings_calendar(datetime)
      )
    )

    for row in earnings_rows:
      try:
        stock_data = utils.data_to_list(
          utils.get_stock_data(row[0]),
          data_keys
        )
        dataset_row = create_earnings_dataset_row(
          stock_data, row[-1], classification_function
        )
        data.append(dataset_row)
      except Exception as e:
        print(e)
  except Exception as e:
    print(e)

  return data


def create_earnings_dataset_range(
  datetime,
  days_ahead,
  data_keys,
  classification_function):
  # get current stock data and results for a range of dates
  data = []

  for i in range(0, days_ahead):
    date = datetime + timedelta(i)
    data.extend(
      create_earnings_dataset_date(date, data_keys, classification_function)
    )

  return data