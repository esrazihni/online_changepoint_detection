import matplotlib.pyplot as plt
import matplotlib.animation as animation
import urllib.request, json
import datetime
from scipy import stats
from bayesian_ocd import *



def get_currency_data(name):
    # https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=BTC&to_currency=USD&apikey=D8N9ZL5YTPIWAD99
    # https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_INTRADAY&symbol=BTC&market=EUR&&apikey=D8N9ZL5YTPIWAD99
    # https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AMZN&interval=60min&apikey=D8N9ZL5YTPIWAD99
    with urllib.request.urlopen('https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_INTRADAY&symbol=%s&market=EUR&&apikey=D8N9ZL5YTPIWAD99' % name) as url:
        data = json.loads(url.read().decode())
        if 'Error Message' in data:
            print('Service is not available')
            dates = []
            x = []
        else:
            dates = []
            x = []
            all_data = list(data['Time Series (Digital Currency Intraday)'].keys())
            for data_point in all_data:
                dates.append(datetime.datetime.strptime(data_point, '%Y-%m-%d %H:%M:%S'))
                #print(data['Time Series (1min)'][data_point]['4. close'])
                x.append(float(data['Time Series (Digital Currency Intraday)'][data_point]['1a. price (EUR)']))
        #x=[float(price)]
        #dates=[datetime.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')]
    return x,dates


def get_stock_data(name):
    # https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AMZN&interval=60min&apikey=D8N9ZL5YTPIWAD99
    #with urllib.request.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=1min&apikey=D8N9ZL5YTPIWAD99' % name) as url:
    with urllib.request.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=1min&apikey=D8N9ZL5YTPIWAD99' % name) as url:
        data = json.loads(url.read().decode())
        if 'Error Message' in data:
            print('Service is not available')
            dates = []
            x = []
        else:
            dates = []
            x = []
            #all_data = list(data['Monthly Time Series'].keys())
            all_data = list(data['Time Series (1min)'].keys())
            for data_point in all_data:
                #dates.append(datetime.datetime.strptime(data_point, '%Y-%m-%d'))
                dates.append(datetime.datetime.strptime(data_point, '%Y-%m-%d %H:%M:%S'))
                #print(data['Time Series (1min)'][data_point]['4. close'])
                #x.append(float(data['Monthly Time Series'][data_point]['4. close']))
                x.append(float(data['Time Series (1min)'][data_point]['4. close']))
        #x=[float(price)]
        #dates=[datetime.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')]
    return x,dates

def plot_data(i):
    #print(prob_r)
    global dates_new
    global x_new
    global last_value_of_last_array

    if(len(x_new)==0):
        if (return_len > 0):
            for x_id, x_date in enumerate(x_dates[0]):
                if(x_id > 0):
                    x_new.append(((x_date/x_dates[0][x_id-1])-1)*700)
            last_value_of_last_array = x_dates[0][return_len-1]
            dates_new = x_dates[1][1:]
    else:
        if(return_len>0):
            if(last_value_of_last_array != x_dates[0][return_len - 1]):
                x_new.append(((x_dates[0][return_len-1]/x_dates[0][return_len-2])-1)*700)
                dates_new.append(x_dates[1][-1])
                last_value_of_last_array = x_dates[0][return_len - 1]

    #print(x_new)

    prob_r = inference(x_new,dist='norm')

    ax1.clear()
    ax2.clear()
    #ax.plot_date(x_dates[1], x_dates[0], markersize=0.5)
    ax1.plot_date(dates_new, x_new,'b-', markersize=0.5)
    ax2.imshow(-np.log(prob_r), interpolation='none', aspect='auto', origin='lower', cmap=plt.cm.Blues)

#AMZN
#MSFT
#AAPL
last_value_of_last_array = 0
stock_symbol='AMZN'
currency_symbol='BTC'

x_dates = get_stock_data(stock_symbol)
#x_dates = get_currency_data(currency_symbol)
return_count = 0
return_len = len(x_dates[0])

x_new = []
dates_new = []

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

#describe_data(get_stock_data(stock_symbol))

#ask for new data every 20 seconds

ani = animation.FuncAnimation(fig, plot_data, interval=10000)
plt.show()