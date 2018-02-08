import matplotlib.pyplot as plt
import matplotlib.animation as animation
import urllib.request, json
import datetime
from scipy import stats
from bayesian_ocd import *


def get_stock_data(name):
    # https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AMZN&interval=60min&apikey=D8N9ZL5YTPIWAD99
    #with urllib.request.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=%s&interval=1min&apikey=D8N9ZL5YTPIWAD99' % name) as url:
    with urllib.request.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol=%s&interval=1min&apikey=D8N9ZL5YTPIWAD99' % name) as url:
        data = json.loads(url.read().decode())
        if 'Error Message' in data:
            print('Service is not available')
            dates = []
            x = []
        else:
            dates = []
            x = []
            all_data = list(data['Monthly Time Series'].keys())
            #all_data = list(data['Time Series (1min)'].keys())
            for data_point in all_data:
                dates.append(datetime.datetime.strptime(data_point, '%Y-%m-%d'))
                #dates.append(datetime.datetime.strptime(data_point, '%Y-%m-%d %H:%M:%S'))
                #print(data['Time Series (1min)'][data_point]['4. close'])
                x.append(float(data['Monthly Time Series'][data_point]['4. close']))
                #x.append(float(data['Time Series (1min)'][data_point]['4. close']))
        #x=[float(price)]
        #dates=[datetime.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')]
    return x,dates

def plot_data(i):
    x_dates = get_stock_data(stock_symbol)
    return_len = len(x_dates[0])
    
    #print(prob_r)
    global dates_new
    global x_new


    if(len(x_new)==0):
        if (return_len > 0):
            for x_id, x_date in enumerate(x_dates[0]):
                if(x_id > 0):
                    x_new.append((x_date/x_dates[0][x_id-1])-1)
            dates_new = x_dates[1][1:]
    else:
        if(return_len>0):
            x_new.append((x_dates[0][return_len-1]/x_dates[0][return_len-2])-1)
            dates_new.append(x_dates[1][-1])

    print(x_new)

    prob_r = inference(x_new)

    ax1.clear()
    ax2.clear()
    #ax.plot_date(x_dates[1], x_dates[0], markersize=0.5)
    ax1.plot_date(dates_new, x_new,'b-', markersize=0.5)
    ax2.imshow(-np.log(prob_r), interpolation='none', aspect='auto', origin='lower', cmap=plt.cm.Blues)

#AMZN
#MSFT
#AAPL
stock_symbol='AMZN'
x_new = []
dates_new = []

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

#describe_data(get_stock_data(stock_symbol))

#ask for new data every 20 seconds

ani = animation.FuncAnimation(fig, plot_data, interval=10000)
plt.show()