import analyst_functions as af
import scrapers as scrapers

from boto3 import resource
from json import dumps, loads
from os import environ

"""
	DEFAULT DATA:
"""
S3 = resource(
	's3',
	aws_access_key_id=environ['AWS_KEY_ID'],
	aws_secret_access_key=environ['AWS_SECRET_ACCESS_KEY']
)
BUCKET_NAME = "companyearningstradingstrategy"
BUCKET = S3.Bucket(BUCKET_NAME)
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
		num_classifications=af.N_CLASSES,
		classification_function=af.CLASSIFICATION_FUNCTION,
		data_keys=af.DATA_KEYS):
			self.MASTER_DATASET = master_filename
			self.TEST_DATASET = test_filename
			self.N_CLASSES = num_classifications
			self.CLASSIFICATION_FUNCTION = classification_function
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

		self.training_set = loads(
			S3.Object(BUCKET_NAME, master_filename)
			.get()['Body']
			.read()
			.decode('utf-8')
		)

		self.test_set = loads(
			S3.Object(BUCKET_NAME, test_filename)
			.get()['Body']
			.read()
			.decode('utf-8')
		)

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

		training_json = dumps(self.training_set)
		test_json = dumps(self.test_set)

		BUCKET.put_object(Key=master_filename, Body=training_json)
		BUCKET.put_object(Key=test_filename, Body=test_json)


	def train(self):
		self.classifier = af.train(
			self.training_set, self.DATA_WIDTH, self.N_CLASSES
		)

		return self.classifier

	def accuracy(self):
		return af.accuracy(self.test_set, self.classifier)

	def test_data(self, stock_data):
		return af.test_data(stock_data, self.classifier)

	def test_symbol(self, symbol):
		return af.test_symbol(symbol, self.DATA_KEYS, self.classifier)

	def test_symbol_raw(self, symbol):
		self.load_datasets()
		self.train()
		return self.test_symbol(symbol)

	def create_earnings_dataset_range(
		self,
		datetime,
		days_ahead=0,
		dataset='MASTER'):
		# dataset should be either 'MASTER' or 'TEST'
		data = af.create_earnings_dataset_range(
			datetime, days_ahead, self.DATA_KEYS, self.CLASSIFICATION_FUNCTION
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
			and target <= max(self.training_set['targets'])
			and target <= max(self.test_set['targets'])
			and target >= min(self.training_set['targets'])
			and target >= min(self.test_set['targets'])):
			try:
				set_data_row = list(map(lambda e: float(e), set_data_row))
				target = int(target)

				if dataset == 'TEST':
					self.test_set['set_data'].append(set_data_row)
					self.test_set['targets'].append(target)
				elif dataset == 'MASTER':
					self.training_set['set_data'].append(set_data_row)
					self.training_set['targets'].append(target)

				a.train()
			except Exception as e:
				print(e)
		else:
			print('set/target outside bounds')