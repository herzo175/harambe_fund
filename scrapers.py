from sys import version_info
from bs4 import BeautifulSoup

if version_info > (3,0):
	from urllib.request import urlopen
else:
	from urllib import urlopen

from yahoo_finance import Share


"""
	Functions to retreive stock information:
"""

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
	# return dictionary of all data for a stock
	url = 'https://finance.yahoo.com/quote/' + symbol
	soup = BeautifulSoup(urlopen(url), 'html.parser')

	rows = list(
		map(
			lambda r: [e.text.strip() for e in r.find_all('td')],
			soup.find_all('tr')
		)
	)

	d = {r[0]: r[1] for r in rows}

	for k in d:
		d[k] = d[k].replace(',', '')

	d['Ask'] = d['Ask'].split(' x ')[0]
	d['Bid'] = d['Bid'].split(' x ')[0]

	# Create daily buy/sell distances in price
	d['Buy Distance Below Daily High'] = (
		str(
			float(d["Day's Range"].split(' - ')[1]) - float(d['Ask'])
		)
	)
	d['Buy Distance Above Daily Low'] = (
		str(
			float(d['Ask']) - float(d["Day's Range"].split(' - ')[0])
		)
	)
	d['Sell Distance Below Daily High'] = (
		str(
			float(d["Day's Range"].split(' - ')[1]) - float(d['Bid'])
		)
	)
	d['Sell Distance Above Daily Low'] = (
		str(
			float(d['Bid']) - float(d["Day's Range"].split(' - ')[0])
		)
	)

	# create yearly buy/sell distances in price
	d['Buy Distance Below Yearly High'] = (
		str(
			float(d["52 Week Range"].split(' - ')[1]) - float(d['Ask'])
		)
	)
	d['Buy Distance Above Yearly Low'] = (
		str(
			float(d['Ask']) - float(d["52 Week Range"].split(' - ')[0])
		)
	)
	d['Sell Distance Below Yearly High'] = (
		str(
			float(d["52 Week Range"].split(' - ')[1]) - float(d['Bid'])
		)
	)
	d['Sell Distance Above Yearly Low'] = (
		str(
			float(d['Bid']) - float(d["52 Week Range"].split(' - ')[0])
		)
	)

	# create buy/sell distances from yearly target
	d['Buy Distance From Yearly Target'] = (
		str(
			float(d['1y Target Est']) - float(d['Ask'])
		)
	)
	d['Sell Distance From Yearly Target'] = (
		str(
			float(d['1y Target Est']) - float(d['Bid'])
		)
	)

	return d


def data_to_list(dictionary, keys):
	return [dictionary[k] for k in keys]


"""
	Data aggregation functions:
"""


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


def calculate_target_frequencies(filename):
	rows = CSV_to_2D_list(filename)

	return calculate_frequencies(rows, lambda r: r[-1])


"""
	CSV handling functions
"""

def add_to_csv(row, filename):
	str_row = list(map(lambda e: str(e), row))
	new_string = ','.join(str_row) + '\n'

	with open(filename, 'a') as f:
		f.write(new_string)
		f.close()

	return new_string

def CSV_to_2D_list(filename):
	f = open(filename, 'r')
	# split up file string into lines
	lines = f.read().split('\n')[:-1]
	f.close()
	# convert rows from strings to arrays
	rows = list(map(lambda l: l.split(','), lines))

	return rows
