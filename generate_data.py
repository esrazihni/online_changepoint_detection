from __future__ import division
import numpy as np
import csv

# Functions for generated data
def generate_normal_time_series(num, minl=50, maxl=1000):
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    mean_all =[]
    var_all = []
    for p in partition:
        mean = np.random.randn()*10
        mean_all.append(mean)
        var = np.abs(np.random.randn()*1) 
        var_all.append(var)
        tdata = np.random.normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return data, partition

def generate_multi_normal_time_series(num, minl=50, maxl=1000):
    # 2 dimensional case:
    data = np.zeros((1,2))
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean1 = np.random.randn()*10
        mean2 = np.random.randn()*10
        cov = np.zeros((2,2))
        cov[0,0] = np.abs(np.random.randn()*1) 
        cov[1,1] = np.abs(np.random.randn()*1) 
        
        tdata = np.random.multivariate_normal([mean1,mean2], cov, p)
        data = np.vstack((data, tdata))
    return data, partition

def generate_poisson_time_series(num, minl=50, maxl=1000):
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for i,p in enumerate(partition):
        lamb = np.abs(np.random.randn())*(i+1)
        print(lamb)
        tdata = np.random.poisson(lamb, p)
        data = np.concatenate((data, tdata))
    return data, partition




# Functions for data read from file
def get_stock(name):
  tmp = np.loadtxt(name, dtype=np.str, delimiter=",")
  size = tmp[1:, 0].size
  data = np.zeros(size)
  for index, item in enumerate(tmp[1:, 1]): 
    data[index] = item[1:-1] 
  return data[::-1]

def make_stock_gaussian(name):
  data = get_stock(name)
  r = np.diff(data) / data[0:(data.size-1)]
  return r




def get_multi_stock(names):
  data = np.zeros(1)
  for name in names:
    td = get_stock(name)
    if data.size == 1:
      data = td
    else:
      data = np.vstack((data,td))
  return data.transpose()

def make_multi_gaussian(names):
  data = np.zeros(1)
  for name in names:
    td = make_stock_gaussian(name)
    if data.size == 1:
      data = td
    else:
      data = np.vstack((data,td))
  return data.transpose()

