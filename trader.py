from analyst_class import Analyst
from datetime import datetime, timedelta
from pytz import timezone
from random import shuffle
from utils import (
  data_to_list,
  get_stock_data,
  get_earnings_calendar,
  load_json_from_S3,
  save_json_to_S3
)


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


def buy_stocks(current_time):
  # check if time and if day is a trading day
  # current_time.hour == 15 and current_time.weekday() < 5
  if True:
    # get associated earnings for that day
    def earnings_filter(row):
      return (
        'After Market Close' in row
        and row[3] != '-'
        and float(row[3]) > 0.1
      )

    earnings_rows = get_earnings_calendar(current_time)
    earnings_rows = list(
      filter(
        earnings_filter, earnings_rows
      )
    )

    # get stock data for each of those earnings
    stock_data = []

    for r in earnings_rows:
      try:
        stock_data.append(
          {
            'date_evaluated': (
              str(current_time.year)
              + '-'
              + str(current_time.month)
              + '-'
              + str(current_time.day)
            ),
            'symbol': r[0],
            'data': data_to_list(get_stock_data(r[0]), A.DATA_KEYS),
            'prediction': A.test_symbol(r[0])
          }
        )
      except Exception as e:
        print(e)

    # save temporary stock data
    add_to_temp_data(stock_data)

    # filter earnings and pick 3 stocks
    picked_stocks = list(
      filter(lambda e: int(e['prediction']) == 3, stock_data)
    )
    shuffle(picked_stocks)
    picked_stocks = picked_stocks[:3]
    # execute trades

    return picked_stocks


def sell_stocks():
  pass


def add_to_temp_data(new_data):
  print(new_data)

  temp_data = load_json_from_S3('temp_data.json')
  temp_data.extend(new_data)
  save_json_to_S3(temp_data, 'temp_data.json')
  pass


def add_to_dataset():
  pass


if __name__ == '__main__':
  current_time = datetime.now(tz=timezone("US/Eastern"))

  buy_stocks(current_time)