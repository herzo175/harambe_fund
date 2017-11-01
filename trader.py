from analyst_class import Analyst
from datetime import datetime, timedelta
from pytz import timezone
from random import shuffle
from schedule import every, run_pending
from utils import (
  BROKERAGE,
  ita,
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

NUM_STOCKS = 3


A = Analyst(
  master_filename="training_set.json",
  test_filename="test_set.json",
  num_classifications=4,
  classification_function=CLASSIFICATION_FUNCTION,
  data_keys=DATA_KEYS
  )


def buy_stocks():
  current_time = datetime.now(tz=timezone("US/Eastern"))
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
      stock_dict = get_stock_data(r[0])

      if (
        float(stock_dict['Beta']) < 1
      ):
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
            'data': data_to_list(stock_dict, A.DATA_KEYS),
            'prediction': A.test_symbol(r[0])
          }
        )
    except Exception as e:
      print(e)

  # save temporary stock data
  add_to_temp_data(stock_data)

  # filter earnings
  picked_stocks = list(
    filter(lambda e: int(e['prediction']) >= 3, stock_data)
  )
  # shuffle(picked_stocks)
  # picked_stocks = picked_stocks[:NUM_STOCKS]

  # execute trades
  portfolio_status = BROKERAGE.get_portfolio_status()
  # partition_size = portfolio_status.cash / NUM_STOCKS
  partition_size = portfolio_status.cash / len(picked_stocks)

  for s in picked_stocks:
    price = ita.get_quote(s['symbol'])
    num_shares = int(partition_size // price)
    print('symbol:', s['symbol'])
    print('price:', price)
    print('num_shares:', num_shares)
    BROKERAGE.trade(s['symbol'], ita.Action.buy, num_shares)

  return picked_stocks


def sell_stocks():
  # sell all stocks
  for s in BROKERAGE.get_current_securities().bought:
    # sell if profit or loss is within 1% of purchase price
    if (
      s.current_price > s.purchase_price
      or (1 - (s.current_price / s.purchase_price)) < .01
      ):
      symbol = s.symbol
      quantity = s.quantity
      BROKERAGE.trade(symbol, ita.Action.sell, quantity)


def add_to_temp_data(new_data):
  print(new_data)

  temp_data = load_json_from_S3('temp_data.json')
  temp_data.extend(new_data)
  save_json_to_S3(temp_data, 'temp_data.json')


def add_to_dataset():
  temp_data = load_json_from_S3('temp_data.json')
  # filter out data more than 7 days old
  def younger_than_ten_days(datestring):
    ds = datestring.split('-')
    dt = datetime(int(ds[0]), int(ds[1]), int(ds[2]))

    if dt + timedelta(10) > datetime.now():
      return True
    else:
      return False

  temp_data = list(
    filter(
      lambda e: younger_than_ten_days(e['date_evaluated']), temp_data
    )
  )

  # group earnings by date
  temp_data_2 = temp_data[:]
  earnings_by_date = {}

  for i, e in enumerate(temp_data_2):
    if e['date_evaluated'] in earnings_by_date and 'result' not in e:
      earnings_by_date[e['date_evaluated']] += [e]
    elif 'result' not in e:
      earnings_by_date[e['date_evaluated']] = [e]
    else:
      # get rid of data that has results
      del(temp_data[i])

  # lookup earnings for each date
  for datestring in earnings_by_date:
    ds = datestring.split('-')
    dt = datetime(int(ds[0]), int(ds[1]), int(ds[2]))
    earnings_by_stock = {e[0]: e[5] for e in get_earnings_calendar(dt)}
    print(earnings_by_date[datestring])
    print(earnings_by_stock)

    for i, e in enumerate(earnings_by_date[datestring]):
      if (
        e['symbol'] in earnings_by_stock
        and earnings_by_stock[e['symbol']] != '-'):
        e['result'] = CLASSIFICATION_FUNCTION(earnings_by_stock[e['symbol']])
        print(e['symbol'], e['result'])
        
        # put every 5th element into the test dataset
        if i % 5 == 0:
          dataset = 'TEST'
        else:
          dataset = 'MASTER'

        A.add_to_dataset(e['data'], e['result'], dataset)
        del(temp_data[temp_data.index(e)])

  A.save_datasets()
  A.train()
  save_json_to_S3(temp_data, 'temp_data.json')

  return temp_data, earnings_by_date


if __name__ == '__main__':
  every().monday.at("10:00", tz=timezone('US/Eastern')).do(sell_stocks)
  every().monday.at("12:00", tz=timezone('US/Eastern')).do(add_to_dataset)
  every().monday.at("14:30", tz=timezone('US/Eastern')).do(buy_stocks)

  every().tuesday.at("10:00", tz=timezone('US/Eastern')).do(sell_stocks)
  every().tuesday.at("12:00", tz=timezone('US/Eastern')).do(add_to_dataset)
  every().tuesday.at("14:30", tz=timezone('US/Eastern')).do(buy_stocks)

  every().wednesday.at("10:00", tz=timezone('US/Eastern')).do(sell_stocks)
  every().wednesday.at("12:00", tz=timezone('US/Eastern')).do(add_to_dataset)
  every().wednesday.at("14:30", tz=timezone('US/Eastern')).do(buy_stocks)

  every().thursday.at("10:00", tz=timezone('US/Eastern')).do(sell_stocks)
  every().thursday.at("12:00", tz=timezone('US/Eastern')).do(add_to_dataset)
  every().thursday.at("14:30", tz=timezone('US/Eastern')).do(buy_stocks)

  every().friday.at("10:00", tz=timezone('US/Eastern')).do(sell_stocks)
  every().friday.at("12:00", tz=timezone('US/Eastern')).do(add_to_dataset)
  every().friday.at("14:30", tz=timezone('US/Eastern')).do(buy_stocks)

  while True:
    run_pending()