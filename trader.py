import analyst_functions

from analyst_class import Analyst
from datetime import datetime, timedelta
from pytz import timezone
from scrapers import data_to_list, get_stock_data


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


A = Analyst(
	master_filename="training_set.json",
	test_filename="test_set.json",
	num_classifications=4,
	classification_function=CLASSIFICATION_FUNCTION,
	data_keys=DATA_KEYS
	)


def buy_stocks():
	# check if time and if day is a trading day
	current_time = datetime.now(tz=timezone("US/Eastern"))

	if current_time.hour == 15 and current_time.weekday() < 5:
		# get day of next trading day
		if current_time.weekday() == 4:
			# handle Fridays
			earnings_date = current_time + timedelta(3)
		else:
			earnings_date = current_time + timedelta(1)

		# get associated earnings for that day
		earnings_rows = get_earnings_calendar(earnings_date)
		# get stock data for each of those earnings
		stock_data = list(
			map(
				lambda r: 
					{
						'date_evaluated': (
							str(current_time.year)
							+ '-'
							+ str(current_time.month)
							+ '-'
							+ str(current_time.day)
						),
						'symbol': r[0],
						'data': data_to_list(get_stock_data(r[0]), A.data_keys),
						'prediction': A.test_symbol(r[0])
					},
					earnings_rows
			)
		)
		# save temporary stock data
		# filter earnings
		# execute trades
	pass


def sell_stocks():
	pass


def extend_data():
	pass


if __name__ == '__main__':
	main()