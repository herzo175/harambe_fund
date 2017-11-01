from os import environ
from sys import version_info
from bs4 import BeautifulSoup
from boto3 import resource
from json import dumps, loads
from InvestopediaApi import ita

if version_info > (3,0):
  from urllib.request import urlopen
else:
  from urllib import urlopen

"""
  Default values across app:
"""
S3 = resource(
  's3',
  aws_access_key_id=environ['AWS_KEY_ID'],
  aws_secret_access_key=environ['AWS_SECRET_ACCESS_KEY']
)
BUCKET_NAME = "companyearningstradingstrategy"
BUCKET = S3.Bucket(BUCKET_NAME)
BROKERAGE = ita.Account(
  environ['INVESTOPEDIA_USERNAME'], environ['INVESTOPEDIA_PASSWORD']
)


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
    (str(date.day) if date.day >= 10 else '0' + str(date.day))
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

  return rows

def get_historical_eps_estimate(symbol):
  url = (
    'https://www.reuters.com/finance/stocks/analyst/' + symbol
  )
  soup = BeautifulSoup(urlopen(url), 'html.parser')
  cols = soup.find_all('td')
  cols = [e.text.strip() for e in cols]

  earnings_starts = []

  for i, r in enumerate(cols):
    if r == 'Earnings (per share)':
      earnings_starts.append(i)

  earnings_cols = cols[earnings_starts[1]+2:earnings_starts[1]+28]

  def is_float(e):
    try:
      float(e)
      return True
    except:
      return False

  earnings_cols = list(filter(lambda r: is_float(r), earnings_cols))

  # [estimate, actual, difference, surprise]
  return [earnings_cols[i:i+4] for i in range(0, len(earnings_cols), 4)]


def get_stock_change(symbol, datetime):
  url = (
    'https://finance.yahoo.com/quote/' + symbol + '/history'
  )
  soup = BeautifulSoup(urlopen(url), 'html.parser')

  rows = soup.find_all('tr')[1:-1]
  rows = list(
    map(lambda r: [e.text.strip() for e in r.find_all('td')], rows)
  )

  def convert_datestring_to_datetime(ds):
    # ex. Jan 25 2017
    dl = ds.replace(',', '').split(' ')

    # month, day, year => year, month, day
    dl[0], dl[1], dl[2] = int(dl[2]), dl[0], int(dl[1])

    if dl[1] == 'Jan':
      dl[1] = 1
    elif dl[1] == 'Feb':
      dl[1] = 2
    elif dl[1] == 'Mar':
      dl[1] = 3
    elif dl[1] == 'Apr':
      dl[1] = 4
    elif dl[1] == 'May':
      dl[1] = 5
    elif dl[1] == 'Jun':
      dl[1] = 6
    elif dl[1] == 'Jul':
      dl[1] = 7
    elif dl[1] == 'Aug':
      dl[1] = 8
    elif dl[1] == 'Sep':
      dl[1] = 9
    elif dl[1] == 'Oct':
      dl[1] = 10
    elif dl[1] == 'Nov':
      dl[1] = 11
    else:
      dl[1] = 12

    return dl

  date_row = list(
    filter(
      lambda r: (
        convert_datestring_to_datetime(r[0])
        ==
        [datetime.year, datetime.month, datetime.day]
      ), rows
    )
  )[0]

  def calculate_difference(open_string, close_string):
    o = float(open_string.replace(',', ''))
    c = float(close_string.replace(',', ''))

    return (c - o) / o

  return calculate_difference(date_row[1], date_row[4])


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
  # mode is either 'a' for append or 'w' for overwrite
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

"""
  AWS Functions
"""

def load_json_from_S3(filename):
  return loads(
    S3.Object(BUCKET_NAME, filename)
    .get()['Body']
    .read()
    .decode('utf-8')
  )

def save_json_to_S3(body, filename):
  BUCKET.put_object(Key=filename, Body=dumps(body))