from __future__ import division
import offline_detection as offcd
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd

#~~~~~~~~~~~~~~~~~~~~~~    
#data = gd.get_multi_stock(['data/apple.csv'])
data = gd.get_multi_stock(['data/microsoft.csv'])
#data = gd.generate_normal_time_series(7, 50, 200)
#data, label = gd.load_from_csv('dowj_1996-2018_with_date.csv')

Q, P, Pcp = offcd.offline_changepoint(data, 
    partial(offcd.const_prior, l=(len(data)+1)), 
    offcd.gaussian_obs_log_likelihood, truncate=-40)

fig, ax = plt.subplots(figsize=[18, 16])
ax = fig.add_subplot(2, 1, 1)
ax.plot(data[:])
ax = fig.add_subplot(2, 1, 2, sharex=ax)
ax.plot(np.exp(Pcp).sum(0))

#~~~~~~~~~~~~~~~~~~~~~~
#plt.savefig('output_images/offline_single_apple.png')
plt.savefig('output_images/offline_single_microsoft.png')
#plt.savefig('output_images/offline_single_example.png')
#plt.savefig('output_images/offline_single_dowj_1996_2018.png')

plt.show()