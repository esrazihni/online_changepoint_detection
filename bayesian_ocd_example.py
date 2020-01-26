import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd
from bayesian_ocd import *

# Getting example data
data_stock = gd.make_multi_gaussian(['data/apple.csv', 'data/microsoft.csv'])

dataset_n, part_n = generate_multi_normal_time_series(4, 150,200)
dataset_n = dataset_n[1:,:]


# Using the algorithm on the datasets
p_stock = inference(dataset_stock,dist='multi')
p_n = inference(dataset_stock,dist='normal')

#Plotting the probability mass
fig, ax = plt.subplots(2,1,figsize=(15,10),sharex=True,gridspec_kw = {'height_ratios':[1.5, 2]})
lab=['microsoft','apple']
for i,l in enumerate(lab):
    ax[0].plot(dataset_stock[:,i] ,label=l)
ax[0].set_title('Apple and Microsoft Stock Data 2008-2013',size=20)
ax[0].set_xlabel('Years',size=14)
ax[0].set_ylabel('Daily return',size=14)
ax[0].legend()
ax1 = ax[1].imshow(p_stock, interpolation='none', aspect='auto',origin='lower', cmap=plt.cm.Blues)
ax[1].imshow(-np.log(p_stock), interpolation='none', aspect='auto',origin='lower', cmap=plt.cm.Blues)
ax[1].set_xlabel('Time points',size=14)
ax[1].set_ylabel('Run length',size=14)
ax[1].set_title('Online Changepoint Detection',size=20)
plt.colorbar(ax1,orientation='horizontal')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2,1,figsize=(15,10),sharex=True,gridspec_kw = {'height_ratios':[1.5, 2]})
ax[0].plot(dataset_n)
for cp in np.cumsum(part_n):
    ax[0].axvline(x=cp, color='r')
ax[0].set_xlim([0,len(dataset_n)])
ax[0].set_title('Generated Normal Data',size=20)
ax[0].set_xlabel('Time points',size=14)
ax[0].set_ylabel('Mean',size=14)
ax[0].legend()
ax1 = ax[1].imshow(p_n, interpolation='none', aspect='auto',origin='lower', cmap=plt.cm.Blues)
ax[1].imshow(-np.log(p_n), interpolation='none', aspect='auto',origin='lower', cmap=plt.cm.Blues)
ax[1].set_xlabel('Time points',size=14)
ax[1].set_ylabel('Run length',size=14)
ax[1].set_title('Online Changepoint Detection',size=20)
plt.colorbar(ax1,orientation='horizontal')
plt.tight_layout()
plt.show()
