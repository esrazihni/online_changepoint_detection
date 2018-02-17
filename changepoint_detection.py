import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm, nbinom
from scipy.special import gamma, gammaln

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
    return data, partition

def generate_poisson_time_series(num, minl=50, maxl=1000):
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for i,p in enumerate(partition):
        lamb = np.abs(np.random.randn())*(i+1)
        #print(lamb)
        tdata = np.random.poisson(lamb, p)
        data = np.concatenate((data, tdata))
    return data, partition

#Read and plot data

DIR_FILENAME = 'dowj_1996-2018.csv'
dataset = np.genfromtxt(DIR_FILENAME, delimiter=',')
#dataset = np.genfromtxt('data/population.csv',delimiter=',')

dataset_n1, part_n1 = generate_normal_time_series(8, 50, 200)
dataset_n2, part_n2 = generate_normal_time_series(8, 50, 200)
#dataset_n  = [dataset_n1[:],dataset_n2[:]]

#dataset_p, part_p = generate_poisson_time_series(8, 1000, 1200)
DIR_FILENAME_POISSON = 'data/crimes_los_angeles_ordered_justnum.csv'
dataset_p = np.genfromtxt(DIR_FILENAME_POISSON, delimiter=',')
print('test a')
print(dataset_p)
print('test b')
# Define functions

def const_hazard(r, lambda_):
    probs = np.ones(r) / lambda_
    return probs


def pred_prob(x, mu, alpha, kappa, beta, mean, var, dist, typ):
    pred = np.zeros(len(alpha), dtype='float128')
    if dist == 'norm':
        if typ == 'both_unknown':
            prec = (alpha * kappa) / (beta * (kappa + 1))
            df = 2 * alpha
            # pred= t.pdf(x, df=df,loc=mu,scale=np.sqrt(1/prec))
            # import pdb; pdb.set_trace()
            g = np.exp(gammaln((df + 1) / 2) - gammaln(df / 2)).astype('float128')
            temp = g * (np.sqrt(prec / (np.pi * df)))
            pred = temp / ((1 + (prec * ((x - mu) ** 2)) / df) ** ((df + 1) / 2))
        elif typ == 'var_known':
            pred = norm.pdf(x, loc=mu, scale=kappa + var)
        elif typ == 'mean_known':
            deg_freedom = 2 * alpha
            precision = alpha / beta
            pred = t.pdf(x, df=deg_freedom, loc=mean, scale=np.sqrt(1 / precision))


    elif dist == 'poisson':
        g = np.exp(gammaln(x + alpha) - (gammaln(alpha) + gammaln(x + 1)))

        pred = g * ((beta / (beta + 1)) ** alpha) * ((1 / (beta + 1)) ** x)

    return pred


def update_stats(x, mu, alpha, kappa, beta, mean, var, dist, typ):
    if dist == 'norm':
        if typ == 'both_unknown':
            mu, alpha, kappa, beta = (kappa * mu + x) / (kappa + 1.), alpha + 0.5, kappa + 1., beta + (
            kappa * ((x - mu) ** 2)) / (2 * (kappa + 1.))
        if typ == 'var_known':
            mu, kappa = mu, kappa
        if typ == 'mean_known':
            alpha, beta = alpha + 0.5, beta + 0.5 * ((x - mean) ** 2)
    if dist == 'poisson':
        alpha, beta = alpha + x, beta + 1.
    return mu, alpha, kappa, beta


def inference(data, dist, mu0=0, kappa0=1, alpha0=1, beta0=1, lam=[500], typ='mean_known', mean=0, var=1):
    # Initializ
    prob_r = np.zeros((len(data) + 1, len(data) + 1))
    prob_r[0, 0] = 1
    mu, kappa, alpha, beta = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])

    # Start iteration
    for t, x in enumerate(data):
        # Calculate predictive probability for new data point
        pred = pred_prob(x, mu, alpha, kappa, beta, mean, var, dist, typ)

        # Calculate growth probability
        prob_r[1:t + 2, t + 1] = prob_r[:t + 1, t] * pred * (1 - const_hazard(t + 1, lam))

        # Calculate changepoint probability
        prob_r[0, t + 1] = np.sum(prob_r[:t + 1, t] * pred * const_hazard(t + 1, lam))

        # Calculate evidence
        sum_prob_r = np.sum(prob_r[:, t + 1])

        # Run length distribution
        prob_r[:, t + 1] = prob_r[:, t + 1] / sum_prob_r

        # Update sufficient statistics
        mu_n, alpha_n, kappa_n, beta_n = update_stats(x, mu, alpha, kappa, beta, mean, var, dist, typ)
        mu, kappa, alpha, beta = np.append(np.array([mu0]), mu_n), np.append(np.array([kappa0]), kappa_n), np.append(
            np.array([alpha0]), alpha_n), np.append(np.array([beta0]), beta_n)
        # print(alpha)
        # print(prob_r)

    return prob_r

p = inference(dataset_p,dist='poisson')

# plot
fig = plt.figure(figsize=(15,6))
plt.plot(dataset_p)
for cp in np.cumsum(part_p):
    plt.axvline(x=cp, color='r')

plt.show()

plt.figure(figsize=[15,6])
plt.imshow(-np.log(p), interpolation='none', aspect='auto',origin='lower', cmap=plt.cm.Blues)

#plt.show()

import matplotlib.cm as cm
plt.pcolor(np.array(range(0, len(p[:,0]))),
          np.array(range(0, len(p[:,0]))),
          -np.log(p),
          cmap=cm.Greys, vmin=0, vmax=30)


fig = plt.figure(figsize=(15,6))
plt.plot(np.exp(p).sum(0))
plt.show()