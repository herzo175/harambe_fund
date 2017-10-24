from sys import version_info

from bs4 import BeautifulSoup
from datetime import datetime, timedelta

if version_info > (3,0):
	from urllib.request import urlopen
else:
	from urllib import urlopen

from yahoo_finance import Share


def get_earnings_calendar(date):
	# return 2D array of company earnings information by date
	# [[symbol, name, release_time, expect, actual, surprise], ...]
	date_string = (
		str(date.year) +
		'-' +
		str(date.month) +
		'-' +
		str(date.day)
	)
	url = (
		'https://finance.yahoo.com/calendar/earnings?day=' +
		date_string
	)
	soup = BeautifulSoup(urlopen(url), 'html.parser')

	rows = soup.find_all('tr')[1:]
	rows = list(
		map(lambda r: [e.text.strip() for e in r.find_all('td')], rows)
	)
	filtered_rows = list(
		filter(
			lambda r: '-' not in r, rows
		)
	)

	return filtered_rows


def get_stock_data(symbol):
	# return array of stock data for a given stock
	data = []
	stock = Share(symbol)

	data.append(stock.get_avg_daily_volume() or '0')
	data.append(stock.get_market_cap()[:-1] or '0')
	data.append(stock.get_earnings_share() or '0')
	data.append(stock.get_year_high() or '0')
	data.append(stock.get_year_low() or '0')
	data.append(stock.get_change_from_50_day_moving_average() or '0')
	data.append(stock.get_dividend_share() or '0')
	data.append(stock.get_price_earnings_ratio() or '0')
	data.append(stock.get_price_earnings_growth_ratio() or '0')
	data.append(stock.get_EPS_estimate_next_quarter() or '0')
	data.append(stock.get_EPS_estimate_next_year() or '0')
	data.append(stock.get_one_yr_target_price() or '0')

	return data


def create_earnings_dataset_row(stock_data, target):
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


def create_earnings_dataset_date(datetime):
	data = []

	try:
		earnings_rows = get_earnings_calendar(datetime)

		for row in earnings_rows:
			try:
				stock_data = get_stock_data(row[0])
				dataset_row = create_earnings_dataset_row(
					stock_data, row[-1]
				)
				data.append(dataset_row)
			except:
				pass
	except:
		pass

	return data


def create_earnings_dataset_range(datetime, days_ahead):
	data = []

	for i in range(0, days_ahead):
		date = datetime + timedelta(i)
		data.extend(create_earnings_dataset_date(date))
		
	return data


def add_to_csv(row, filename):
	new_string = ','.join(row) + '\n'
	
	with open(filename, 'a') as f:
		f.write(new_string)
		f.close()

	return new_string

def calculate_frequencies(arr, extractor):
	# extractor is a function that takes an array
	# and returns an element from the array
	targets = list(map(extractor, arr))
	frequencies = {}

	for e in targets:
		if e in frequencies:
			frequencies[e] += 1
		else:
			frequencies[e] = 1

	return frequencies

def CSV_to_2D_list(filename):
	f = open(filename, 'r')
	# split up file string into lines
	lines = f.read().split('\n')[1:-1]
	f.close()
	# convert rows from strings to arrays
	rows = list(map(lambda l: l.split(','), lines))

	return rows


def calculate_target_frequencies(filename):
	rows = CSV_to_2D_list(filename)

	return calculate_frequencies(lasts, lambda r: r[-1])