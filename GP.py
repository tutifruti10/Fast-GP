import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from operator import itemgetter
from scipy.optimize import fmin_l_bfgs_b

class GP():
	def __init__(self, kernel_, mean=None, opt="fmin_l_bfgs_b",n_rest_opt = 0):
		self.kernel_ = kernel_
		self.mean = mean
		self.opt=opt
		self.n_rest_opt=n_rest_opt
		self.K = None
		self.Xtrain = None
		self.Ytrain = None
		self.trained=False
		self.L_trained = None
		self.params=None
		
		
	def add_training(self,Xtrain,Ytrain,params):
		self.Xtrain = Xtrain
		self.Ytrain=Ytrain
		self.trained=True
		self.params=params
		theta_in=self.params[0][0]
		sig=self.params[0][1]
		bounds=self.params[1]
		#fix function 
		def  fun_opt(self,theta=None):
				sig=1
				par=np.array((theta,sig))
				self.K =  self.kernel.get_kernel(self.kernel_(np.array([0.6, 1. ])),Xtrain,Xtrain)
				L=np.linalg.cholesky(self.K + 1e-6*np.eye(Xtrain.shape[0]))
				lml=-0.5*np.dot(np.linalg.solve(L,Ytrain).T,np.linalg.solve(L,Ytrain))-np.sum(np.log(np.diagonal(L)),axis=0)-(Xtrain.shape[0]/2)*np.log(2*np.pi)
				return -lml
		
		theta_opt, func_min, convergence_dict = fmin_l_bfgs_b(fun_opt,theta_in, bounds)
		lml_values = list(map(itemgetter(1), theta_opt))
		self.params[0][0] = theta_opt[np.argmin(lml_values)]
		return self.params[0][0]
          #  self.log_marginal_likelihood_value_ = -np.min(lml_values)

	def predict(self,Xpredict,params,plot=True, n=3):
		kernel_ = self.kernel_
		K_ss = self.kernel.get_kernel(self.kernel_(*params),Xpredict,Xpredict)
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
			Ks = self.kernel.get_kernel(self.kernel_(*params),self.Xtrain,Xpredict)
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

        
class kernel():
	def __init__(self,params,params_bounds=None):
				self.params=params
				self.params_bounds=params_bounds
	def get_par(self):
		return self.params
	def get_bounds(self):
		return self.params_bounds
	def theta(self):
			return self.params[0]
				
class sqexp(kernel):
	def __init__(self,params,params_bounds=None):
		self.params=params
		self.params_bounds=params_bounds
		self.a=None
		self.b=None
		self.l=None
		self.sig=None
		self.hp=None
	def get_kernel(self,a,b=None):
		self.a=a	
		self.b=b
		self.l=self.params[0]
		self.sig=self.params[1]
		if b is None:
			sqdist = pdist(a/self.l,metric='sqeuclidean')
			return self.sig*squareform(np.exp(-.5*sqdist))
		else:
			sqdist=cdist(a/self.l,b/self.l,metric='sqeuclidean')
			return self.sig*np.exp(-0.5 * sqdist)
