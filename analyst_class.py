import analyst_functions as af
import scrapers

"""
	DEFAULT DATA:
"""

MASTER_DATASET = "master.csv"
TEST_DATASET = "test.csv"

"""
	CLASS
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
		master_filename=None,
		test_filename=None):

		master_filename = (
			self.MASTER_DATASET if master_filename is None else master_filename
		)
		test_filename = (
			self.TEST_DATASET if test_filename is None else test_filename
		)

		for i, r in enumerate(self.training_set['set_data']):
			row = r + [self.training_set['targets'][i]]
			scrapers.add_to_csv(row, master_filename)

		for i, r in enumerate(self.test_set['set_data']):
			row = r + [self.test_set['targets'][i]]
			scrapers.add_to_csv(row, test_filename)


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