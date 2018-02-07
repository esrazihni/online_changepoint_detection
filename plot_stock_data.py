import matplotlib.pyplot as plt
import matplotlib.animation as animation
import urllib.request, json
import time
import datetime
from scipy import stats



def get_stock_data(name):
    # https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AMZN&interval=60min&apikey=D8N9ZL5YTPIWAD99
    with urllib.request.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=1min&apikey=D8N9ZL5YTPIWAD99' % name) as url:
        data = json.loads(url.read().decode())
        if 'Error Message' in data:
            print('Service is not available')
            dates = []
            x = []
        else:
            dates = []
            x = []
            all_data = list(data['Time Series (1min)'].keys())
            for data_point in all_data:
                dates.append(datetime.datetime.strptime(data_point, '%Y-%m-%d %H:%M:%S'))
                #print(data['Time Series (1min)'][data_point]['4. close'])
                x.append(float(data['Time Series (1min)'][data_point]['4. close']))
        #x=[float(price)]
        #dates=[datetime.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')]
    return x,dates



def describe_data(x):

    N = len(x)
    #print(x[0])
    #print(x[1])

    #to get last element
    #print(x[0][N])
    #print(x[1][N])

    print(stats.mstats.normaltest(x[0]))
    print(stats.mstats.chisquare(x[0]))
    print(stats.describe(x[0]))


def plot_data(i):
    x_dates = get_stock_data(stock_symbol)
    ax.clear()
    ax.plot_date(x_dates[1], x_dates[0], markersize=0.5)


stock_symbol='AMZN'

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

describe_data(get_stock_data(stock_symbol))

#ask for new data every 20 seconds
ani = animation.FuncAnimation(fig, plot_data, interval=20000)
plt.show()

