from numpy import genfromtxt
import datetime
import matplotlib.pyplot as plt

DIR_FILENAME = 'dowj_1996-2018_with_date.csv'

timeconverter = lambda x_date: datetime.datetime.strptime(x_date.decode("utf-8"), '%Y-%m-%d')
byteconverter = lambda x_data: x_data

dates = genfromtxt(DIR_FILENAME, delimiter=',', converters={0:timeconverter},usecols=(0)).transpose()

print(dates)
x = genfromtxt(DIR_FILENAME, delimiter=',',usecols=(1)).transpose()
# x = x[]
changepoints = []
N = x.size
print(dates)
print(x)
print(N)

# plot
figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot_date(dates,x,markersize=0.5)

plt.show()