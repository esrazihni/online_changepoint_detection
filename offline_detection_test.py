from __future__ import division
import offline_detection as offcd
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd

#~~~~~~~~~~~~~~~~~~~~~~    
#data = gd.get_multi_stock(['data/apple.csv'])
data = gd.get_multi_stock(['data/microsoft.csv'])
#data = gd.make_multi_gaussian(['data/microsoft.csv'])
#data = gd.generate_normal_time_series(7, 50, 200)

Q, P, Pcp = offcd.offline_changepoint(data, 
    partial(offcd.const_prior, l=(len(data)+1)), 
    offcd.gaussian_obs_log_likelihood, truncate=-40)

fig, ax = plt.subplots(figsize=[18, 16])
plt.subplot(211)
plt.plot(data[:])
plt.subplot(212)
plt.plot(np.exp(Pcp).sum(0))

#~~~~~~~~~~~~~~~~~~~~~~
#plt.savefig('output_images/offline_single_apple.png')
plt.savefig('output_images/offline_single_microsoft.png')
#plt.savefig('output_images/offline_single_apple_gaussian.png')
#plt.savefig('output_images/offline_single_microsoft_gaussian.png')
#plt.savefig('output_images/offline_single_example.png')

plt.show()