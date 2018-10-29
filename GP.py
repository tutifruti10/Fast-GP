import numpy as np
import matplotlib.pyplot as plt

class GP():
	def __init__(self, kernel, mean=None):
		self.kernel = kernel
		self.mean = mean
		self.K = None
		self.Xtrain = None
		self.Ytrain = None
		self.trained=False
		self.L_trained = None
		self.params=None
		
		
	def add_training(self,Xtrain,Ytrain,params):
		self.K = self.kernel(Xtrain,Xtrain,*params)
		self.Xtrain = Xtrain
		self.Ytrain=Ytrain
		self.trained=True
		L = np.linalg.cholesky(self.K + 1e-6*np.eye(Xtrain.shape[0]))
		self.L_trained = L
		self.params = params
		return L
		
	def predict(self,Xpredict,params,plot=True, n=3):
		kernel = self.kernel
		K_ss = kernel(Xpredict,Xpredict,*params)
		if not self.trained:
			L_ss = np.linalg.cholesky(K_ss + 0.00005*np.eye(Xpredict.shape[0]))
			f_prior = np.dot(L_ss, np.random.normal(size=(Xpredict.shape[0],n)))
			if plot:
				plt.plot(Xpredict,f_prior)
				plt.show()
		else:
			if self.L_trained is None:
				raise ValueError('GP not trained; please use the add_training function to train it')
			else:
				Lt = self.L_trained
			Ks = kernel(self.Xtrain,Xpredict,*params)
			Ls = np.linalg.solve(Lt,Ks)	
			mu=np.dot(Ls.T, np.linalg.solve(Lt,self.Ytrain)).reshape((Xpredict.shape[0],))
			
			s2 = np.diag(K_ss)-np.sum(Ls**2, axis=0)
			stdv = np.sqrt(s2)
			
			L = np.linalg.cholesky(K_ss + 1e-6*np.eye(Xpredict.shape[0])- np.dot(Ls.T,Ls))
			f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(Xpredict.shape[0],3)))
			
			if plot:
				plt.plot(self.Xtrain,self.Ytrain,'bs',ms=8)
				plt.plot(Xpredict,f_post)
				plt.gca().fill_between(Xpredict.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
				plt.plot(Xpredict,mu,'r--',lw=2)
				plt.show()
	
		
		
		
def sqexp(a,b,l=0.5,sig=1):
	sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a,b.T)
	return sig*np.exp(-0.5 * (1/l) * sqdist)
