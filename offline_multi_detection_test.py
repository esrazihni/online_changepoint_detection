from __future__ import division
import matplotlib.pyplot as plt
import offline_detection as offcd
import numpy as np
from functools import partial
import generate_data as gd


#data, partition = gd.generate_multi_normal_time_series(3,200,500)
data = gd.get_multi_stock(['data/apple.csv', 'data/microsoft.csv'])
partition = np.zeros(0)

fig, ax = plt.subplots(figsize=[18, 8])
plt.subplot(211)
if partition.size != 0:
	changes = np.cumsum(partition)
	for p in changes:
	  plt.plot([p,p],[np.min(data),np.max(data)],'r')
for d in range(2):
  plt.plot(data[:,d])
plt.legend(['Raw data'])

Q_full, P_full, Pcp_full = offcd.offline_changepoint(data,
  partial(offcd.const_prior, l=(len(data)+1)),
  offcd.fullcov_obs_log_likelihood, truncate=-20)


plt.subplot(212)
plt.plot(np.exp(Pcp_full).sum(0))
plt.legend(['Full Covariance Model'])

#plt.savefig('output_images/multi_example.png')
plt.savefig('output_images/multi_stock_apple_microsoft.png')

plt.show()