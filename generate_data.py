from __future__ import division
import numpy as np
import csv

def generate_normal_time_series(num, minl=50, maxl=1000):
  data = np.array([], dtype=np.float64)
  partition = np.random.randint(minl, maxl, num)
  for p in partition:
      mean = np.random.randn()*10
      var = np.random.randn()*1
      if var < 0:
          var = var * -1
      tdata = np.random.normal(mean, var, p)
      data = np.concatenate((data, tdata))
  return data

def generate_multinormal_time_series(num, dim, minl=50, maxl=1000):
  data = np.empty((1,dim), dtype=np.float64)
  partition = np.random.randint(minl, maxl, num)
  for p in partition:
    mean = np.random.standard_normal(dim)*10
    # Generate a random SPD matrix
    A = np.random.standard_normal((dim,dim))
    var = np.dot(A,A.T)

    tdata = np.random.multivariate_normal(mean, var, p)
    data = np.concatenate((data, tdata))
  return partition, data[1:,:]

def generate_multivariate_example(minl=50, maxl=1000):
  dim = 2
  num = 3
  partition = np.random.randint(minl, maxl, num)
  mu = np.zeros(dim)
  Sigma1 = np.asarray([[1.0,0.75],[0.75,1.0]])
  data = np.random.multivariate_normal(mu, Sigma1, partition[0])
  Sigma2 = np.asarray([[1.0,0.0],[0.0,1.0]])
  data = np.concatenate((data,np.random.multivariate_normal(mu, Sigma2, partition[1])))
  Sigma3 = np.asarray([[1.0,-0.75],[-0.75,1.0]])
  data = np.concatenate((data,np.random.multivariate_normal(mu, Sigma3, partition[2])))
  return data, partition

def load_from_csv(filename):
  tmp = np.loadtxt(filename, dtype=np.str, delimiter=",")
  data = tmp[1:,1].astype(np.float)
  label = tmp[0:,0]
  return data, label

def get_stock(name):
  tmp = np.loadtxt(name, dtype=np.str, delimiter=",")
  size = tmp[1:, 0].size
  data = np.zeros(size)
  for index, item in enumerate(tmp[1:, 1]): 
    data[index] = item[1:-1] 
  return data[::-1]

def get_multi_stock(names):
  data = np.zeros(1)
  for name in names:
    td = get_stock(name)
    if data.size == 1:
      data = td
    else:
      data = np.vstack((data,td))
  return data.transpose()

# file name with .csv
# daily return
def make_stock_gaussian(name):
  data = get_stock(name)
  r = np.diff(data) / data[0:(data.size-1)]
  return r

def make_multi_gaussian(names):
  data = np.zeros(1)
  for name in names:
    td = make_stock_gaussian(name)
    if data.size == 1:
      data = td
    else:
      data = np.vstack((data,td))
  return data.transpose()

