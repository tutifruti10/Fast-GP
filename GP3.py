import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from operator import itemgetter
from scipy.optimize import fmin_l_bfgs_b
import copy

class GP():
	def __init__(self, kernel, mean=None, opt="fmin_l_bfgs_b",n_rest_opt = 0):
		self.kernel= kernel
		self.mean = mean
		self.opt=opt
		self.n_rest_opt=n_rest_opt
		self.K = None
		self.Xtrain = None
		self.Ytrain = None
		self.trained=False
		self.L_trained = None
		self.par=None
		self.bounds=None
		
	def add_training(self,Xtrain,Ytrain,par,bounds):
		self.Xtrain = Xtrain
		self.Ytrain=Ytrain
		self.trained=True
		self.par=par
		theta_in=self.par[0]
		self.bounds=bounds
		def lml(*theta):
				K=self.kernel.get_kernel(Xtrain,Xtrain,*theta)
				L=np.linalg.cholesky(K + 1e-6*np.eye(self.Xtrain.shape[0]))
				log_like=-0.5*np.dot(np.linalg.solve(L,self.Ytrain).T,np.linalg.solve(L,self.Ytrain))-np.sum(np.log(np.diagonal(L)),axis=0)-(self.Xtrain.shape[0]/2)*np.log(2*np.pi)
				return -log_like
		optima= [fmin_l_bfgs_b(lml,x0=par, approx_grad=True,bounds=self.bounds)]
		bounds=np.array(self.bounds)
		for iteration in range(self.n_rest_opt):
                    theta_initial =\
                        np.random.uniform(bounds[:,0],bounds[:,1])
                    optima.append(fmin_l_bfgs_b(lml,x0=theta_initial, approx_grad=True,bounds=self.bounds))
		lml_values = list(map(itemgetter(1),optima))
		opt= optima[np.argmin(lml_values)][0]
		self.par= opt
		self.K = self.kernel.get_kernel(Xtrain,Xtrain,*self.par)
		L = np.linalg.cholesky(self.K + 1e-6*np.eye(Xtrain.shape[0]))
		self.L_trained = L
		return L
          #  self.log_marginal_likelihood_value_ = -np.min(lml_values)

	def predict(self,Xpredict,plot=True, n=3):
		K_ss = self.kernel.get_kernel(Xpredict,Xpredict,*self.par)
		if not self.trained:
			L_ss = np.linalg.cholesky(K_ss + 0.00005*np.eye(Xpredict.shape[0]))
			f_prior = np.dot(L_ss, np.random.normal(size=(Xpredict.shape[0],n)))
			if plot:
				plt.plot(Xpredict,f_prior)
				plt.show()
			return f_prior
		else:
			if self.L_trained is None:
				raise ValueError('GP not trained; please use the add_training function to train it')
			else:
				Lt = self.L_trained
			Ks = self.kernel.get_kernel(self.Xtrain,Xpredict,*self.par)
			Ls = np.linalg.solve(Lt,Ks)	
			mu=np.dot(Ls.T, np.linalg.solve(Lt,self.Ytrain)).reshape((Xpredict.shape[0],))
			
			s2 = np.diag(K_ss)-np.sum(Ls**2, axis=0)
			stdv = np.sqrt(s2)
			
			L = np.linalg.cholesky(K_ss + 1e-6*np.eye(Xpredict.shape[0])- np.dot(Ls.T,Ls))
			f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(Xpredict.shape[0],n)))
			
			if plot:
				plt.plot(self.Xtrain,self.Ytrain,'bs',ms=8)
				plt.plot(Xpredict,f_post)
				plt.gca().fill_between(Xpredict.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
				plt.plot(Xpredict,mu,'r--',lw=2)
				plt.show()
			
			return f_post
"""		
def sqexp(a,b=None,l=0.5,sig=1):
	if b is None:
		sqdist = pdist(a/l,metric='sqeuclidean')
		return sig*squareform(np.exp(-.5*sqdist))
	else:
		sqdist=cdist(a/l,b/l,metric='sqeuclidean')
		return sig*np.exp(-0.5 * sqdist)
"""
class Kernel():
	def __init__(self,params,params_bounds=None):
			self.params=params
			self.params_bounds=params_bounds
	def get_par(self):
		return self.params
	def get_bounds(self):
		return self.params_bounds
	def theta(self):
			return self.params[0]
				
class sqexp(Kernel):
	def __init__(self,params,params_bounds=None):
			self.params_list=['len','sig']
	def get_kernel(a,b=None,l=0.5,sig=1):
		if b is None:
			sqdist = pdist(a/l,metric='sqeuclidean')
			return sig*squareform(np.exp(-.5*sqdist))
		else:
			sqdist=cdist(a/l,b/l,metric='sqeuclidean')
			return sig*np.exp(-0.5 * sqdist)