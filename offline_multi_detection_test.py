from __future__ import division
import matplotlib.pyplot as plt
import offline_detection as offcd
import seaborn
import numpy as np
from functools import partial
import generate_data as gd

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


#data, partition = gd.generate_multivariate_example(200,500)
data = gd.get_multi_stock(['data/apple.csv', 'data/microsoft.csv'])
partition = None

Q_full, P_full, Pcp_full = offcd.offline_changepoint(data,
  partial(offcd.const_prior, l=(len(data)+1)),
  offcd.fullcov_obs_log_likelihood, truncate=-20)

fig, ax = plt.subplots(figsize=[18, 8])
ax = fig.add_subplot(2, 1, 1)
if partition != None:
	changes = np.cumsum(partition)
	for p in changes:
	  ax.plot([p,p],[np.min(data),np.max(data)],'r')
for d in range(2):
  ax.plot(data[:,d])
plt.legend(['Raw data'])

ax = fig.add_subplot(2, 1, 2, sharex=ax)
ax.plot(np.exp(Pcp_full).sum(0))
plt.legend(['Full Covariance Model'])

#plt.savefig('output_images/multi_example.png')
plt.savefig('output_images/multi_stock_apple_microsoft.png')

plt.show()