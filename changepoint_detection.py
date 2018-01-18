from numpy import genfromtxt
import matplotlib.pyplot as pyplot

        
DIR_FILENAME = 'test_data.csv'
x = genfromtxt(DIR_FILENAME, delimiter=',')
#x = x[]
changepoints = []
N = x.size
print(x)
print(N)

# plot
figure = pyplot.figure()
ax = figure.add_subplot(2,1,1)
ax.plot(x)

pyplot.show()