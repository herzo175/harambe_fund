from analyst_functions import (
  N_CLASSES,
  TARGET_FROM_EARNINGS_ROW_FUNC,
  CLASSIFICATION_FUNCTION,
  DATA_KEYS,
  train,
  accuracy,
  test_data,
  test_symbol,
  create_earnings_dataset_range
)
from os import environ
from utils import load_json_from_S3, save_json_to_S3

"""
  DEFAULT DATA:
"""
MASTER_DATASET = "training_set.json"
TEST_DATASET = "test_set.json"

"""
  CLASS (used to more easily store )
"""

class Analyst():
  def __init__(
    self,
    master_filename=MASTER_DATASET,
    test_filename=TEST_DATASET,
    num_classifications=N_CLASSES,
    target_from_earnings_row_func=TARGET_FROM_EARNINGS_ROW_FUNC,
    classification_function=CLASSIFICATION_FUNCTION,
    data_keys=DATA_KEYS):
      self.MASTER_DATASET = master_filename
      self.TEST_DATASET = test_filename
      self.N_CLASSES = num_classifications
      self.TARGET_FROM_EARNINGS_ROW_FUNC = target_from_earnings_row_func
      self.CLASSIFICATION_FUNCTION = classification_function
      self.DATA_KEYS = data_keys

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

    self.training_set = load_json_from_S3(master_filename)
    self.test_set = load_json_from_S3(test_filename)

    return self.training_set, self.test_set

  def save_datasets(
    self,
    master_filename=None,
    test_filename=None):

    master_filename = (
      self.MASTER_DATASET if master_filename is None else master_filename
    )
    test_filename = (
      self.TEST_DATASET if test_filename is None else test_filename
    )

    save_json_to_S3(self.training_set, master_filename)
    save_json_to_S3(self.test_set, test_filename)


  def train(self):
    self.classifier = train(
      self.training_set, self.N_CLASSES
    )

    return self.classifier

  def accuracy(self):
    return accuracy(
      self.test_set, self.TARGET_FROM_EARNINGS_ROW_FUNC, self.classifier
    )

  def test_data(self, stock_data):
    return test_data(stock_data, self.classifier)

  def test_symbol(self, symbol):
    return test_symbol(symbol, self.DATA_KEYS, self.classifier)

  def test_symbol_raw(self, symbol):
    self.load_datasets()
    self.train()
    return self.test_symbol(symbol)

  def create_earnings_dataset_range(
    self,
    datetime,
    days_ahead=1,
    dataset='MASTER'):
    # dataset should be either 'MASTER' or 'TEST'
    data = create_earnings_dataset_range(
      datetime,
      days_ahead,
      self.DATA_KEYS,
      self.TARGET_FROM_EARNINGS_ROW_FUNC,
      self.CLASSIFICATION_FUNCTION
    )

    set_data = list(map(lambda r: r[:-1], data))
    targets = list(map(lambda r: r[-1], data))

    if dataset == 'TEST':
      self.test_set['set_data'].extend(set_data)
      self.test_set['targets'].extend(targets)
    elif dataset == 'MASTER':
      self.training_set['set_data'].extend(set_data)
      self.training_set['targets'].extend(targets)

    return data


  def add_to_dataset(self, set_data_row, target, dataset='MASTER'):
    if (
      len(set_data_row) == len(self.training_set['set_data'][0])
      and len(set_data_row) == len(self.test_set['set_data'][0])
      and int(target) <= max(self.training_set['targets'])
      and int(target) <= max(self.test_set['targets'])
      and int(target) >= min(self.training_set['targets'])
      and int(target) >= min(self.test_set['targets'])):
      try:
        set_data_row = list(map(lambda e: float(e), set_data_row))
        target = int(target)

        if dataset == 'TEST':
          self.test_set['set_data'].append(set_data_row)
          self.test_set['targets'].append(target)
        elif dataset == 'MASTER':
          self.training_set['set_data'].append(set_data_row)
          self.training_set['targets'].append(target)

        self.train()
      except Exception as e:
        print(e)
    else:
      print('set/target outside bounds')