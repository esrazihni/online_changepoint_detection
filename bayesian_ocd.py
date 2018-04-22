import numpy as np
from scipy.stats import t, norm, nbinom
from scipy.special import gamma, gammaln



def const_hazard(r,lambda_):
	'''Calculate constant hazard function

	Parameters
	----------
	r:  int
		current run length 
	lambda_:  int
		expected average run length

	Returns
	-------
	probs: int
		hazard value	
	'''
    probs = np.ones(r)/lambda_
    return probs

def pred_prob(x,mu,alpha,kappa,beta,dist):
	'''Calculate the predictive probability

	Parameters
	----------
	x:  float
		new data point 
	mu, alpha, kappa, beta: float
		hyperparameters 
	dist: 'norm','poisson' or 'multi'
		type of the data distribution

	Returns
	-------
	pred: prob
		predictive probability	
	'''
    pred = np.zeros(len(alpha),dtype='float128')
    length = len(kappa)
    if dist=='norm':
        prec = (alpha*kappa)/(beta*(kappa+1))
        df = 2*alpha
        g = np.exp(gammaln((df+1)/2)-gammaln(df/2)).astype('float128')
        temp = g*(np.sqrt(prec/(np.pi*df)))
        pred = temp / ((1+(prec*((x-mu)**2))/df)**((df+1)/2))
        
    elif dist=='poisson':  
        g = np.exp(gammaln(x+alpha).astype('float128') - (gammaln(alpha)+ gammaln(x+1)).astype('float128'))      
        pred = g* ((beta/(beta+1))**alpha)*((1/(beta+1))**x)
        
    elif dist=='multi':
        
        mult = np.zeros(length)
        scale = np.zeros((length,2,2))
        if length == 1 :
            scale = np.linalg.inv((beta*(kappa+1))/(alpha*kappa))
            mult=np.matmul(np.matmul((x-mu),scale),np.transpose(x-mu))
        else :
            scaling = ((kappa+1) / (alpha*kappa))
            for i in range(length):
                scale[i]=np.linalg.inv(beta[i] * scaling[i])
                mult[i]=np.dot(np.dot((x-mu)[i],scale[i]),np.transpose((x-mu)[i]))      
        (sign, logdet) = np.linalg.slogdet(scale)
        
        logc = gammaln(alpha + 1) - gammaln(alpha) + 0.5*logdet.reshape(length,1) - np.log(2*alpha*np.pi)
        pred = np.exp(logc - (alpha + 1)*np.log1p(mult.reshape(len(kappa),1)/(2*alpha)))

        
    return pred.reshape(length,)

def update_stats(x,mu,alpha,kappa,beta,dist):
	'''Calculate the updated sufficient statistics

	Parameters
	----------
	x:  float
		new data point 
	mu, alpha, kappa, beta: float
		hyperparameters 
	dist: 'norm','poisson' or 'multi'
		type of the data distribution

	Returns
	-------
	mu, alpha, kappa, beta: float
		updated hyperparameters 	
	'''
    if dist =='norm':
        mu, alpha, kappa, beta = (kappa*mu + x) / (kappa+1.), alpha+0.5, kappa+1., beta +(kappa*((x-mu)**2))/(2*(kappa+1.))
       
    elif dist == 'poisson':
        alpha, beta = alpha + x , beta+1.
    
    elif dist == 'multi':
        length = len(kappa)
        mu, alpha, kappa = (kappa*mu + x) / (kappa+1.), alpha+0.5, kappa+1.
        for i in range(len(kappa)):
            beta[i] =  beta[i]+ kappa[i]*(np.matmul(np.transpose([(x-mu)[i]]), ([(x-mu)[i]])) / (2. * (kappa[i] + 1.)))
    return mu, alpha, kappa, beta

def inference(data, dist,mu0 = 0, kappa0 = 1, alpha0= 1, beta0= 1, lam = [150]):
	'''Calculate the run length probability

	Parameters
	----------
	data:  array-like
		whole dataset to be used  
	mu0, alpha0, kappa0, beta0: float
		initial hyperparameters 
	lam : int
		initial lambda_
	dist: 'norm','poisson' or 'multi'
		type of the data distribution

	Returns
	-------
	pred: prob_r
		run length probability
	'''
    #Initialize
    prob_r = np.zeros((len(data) + 1, len(data) + 1))
    prob_r[0, 0] = 1
    if dist=='multi':
        mu, kappa, alpha, beta = np.array([mu0,mu0]), np.array([kappa0]), np.array([alpha0]), np.eye(2).reshape(1,2,2)
    else:
        mu, kappa, alpha, beta = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    
    
    #Start iteration   
    for t,x in enumerate(data):  
        #Calculate predictive probability for new data point
        pred = pred_prob(x,mu,alpha,kappa,beta,dist)

        #Calculate growth probability 
        prob_r[1:t+2,t+1] = prob_r[:t+1,t]* pred * (1-const_hazard(t+1,lam))

        #Calculate changepoint probability
        prob_r[0,t+1] = np.sum( prob_r[:t+1,t] * pred * const_hazard(t+1,lam))
        
        #Calculate evidence
        sum_prob_r =  np.sum(prob_r[:,t+1])
        
        #Run length distribution
        prob_r[:,t+1] = prob_r[:,t+1]/sum_prob_r
        
        #Update sufficient statistics
        mu_n, alpha_n, kappa_n, beta_n = update_stats(x, mu, alpha, kappa,beta,dist) 
        mu, kappa, alpha, beta = np.vstack((np.array([0,0]), mu_n)), np.vstack((np.array([1]), kappa_n)), np.vstack((np.array([1]), alpha_n)), np.concatenate((np.eye(2).reshape(1,2,2),beta_n))
        
    return prob_r
