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
		self.params=None
		self.bounds=None
		
	def add_training(self,Xtrain,Ytrain,par,bounds):
		self.Xtrain = Xtrain
		self.Ytrain=Ytrain
		self.trained=True
		self.par=params
		theta_in=self.params[0]
		self.bounds=bounds
		theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(self.lml,x0=0.1, approx_grad=True,bounds=self.bounds)
		lml_values = list(map(itemgetter(1),theta_opt))
		opt= theta_opt[np.argmin(lml_values)]
		return opt
          #  self.log_marginal_likelihood_value_ = -np.min(lml_values)

	def predict(self,Xpredict,params,plot=True, n=3):
		kernel_ = self.kernel
		K_ss = self.kernel.get_kernel(self.kernel(*params),Xpredict,Xpredict)
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
			Ks = self.kernel.get_kernel(self.kernel(*par),self.Xtrain,Xpredict)
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
			
	def lml(self,theta):
				sig=1
				kern = copy.deepcopy(self.kernel)
				kern.params=np.array((theta,sig))
				K =  kern.get_kernel(self.Xtrain,self.Xtrain)
				L=np.linalg.cholesky(K + 1e-6*np.eye(self.Xtrain.shape[0]))
				log_like=0.5*np.dot(np.linalg.solve(L,self.Ytrain).T,np.linalg.solve(L,self.Ytrain))-np.sum(np.log(np.diagonal(L)),axis=0)-(self.Xtrain.shape[0]/2)*np.log(2*np.pi)
				return log_like
        
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
			self.params_list=['sig','len']
	def get_kernel(self,a,b=None):
		if b is None:
			sqdist = pdist(a/self.params[0],metric='sqeuclidean')
			returnself.params[1]*squareform(np.exp(-.5*sqdist))
		else:
			sqdist=cdist(a/self.params[0],b/self.params[0],metric='sqeuclidean')
			return params[1]**np.exp(-0.5 * sqdist)
