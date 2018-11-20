import GP as gp
import numpy as np
import matplotlib.pyplot as plt

Xtrain=np.array((-3,-2,-1,1,2,3)).reshape(6,1)
Ytrain=(np.sin(Xtrain) - np.exp(-1*((Xtrain - 2)**2)/0.5))

Xtrain2=np.array((-3,-2,-1,1,3)).reshape(5,1)
Ytrain2=(np.sin(Xtrain2) - np.exp(-1*((Xtrain2 - 2)**2)/0.5))

n=100
step=10/n

Xpredict=np.mgrid[-5:5:step].reshape(1,n).T
Ypredict=np.sin(Xpredict) - np.exp(-1*((Xpredict - 2)**2)/0.5)

params=np.array((1,1))
bounds=np.array(((0.001,5),(0.001,5)))

compparams=np.array(((1.,1.),(1.,1.)))
compbounds=np.array(((0.001,5),(0.001,5),(0.001,1),(0.001,1)))

kernels=[gp.sqexp,gp.Comp_Kernel(gp.sqexp,gp.sqexp,'kern_sum'),gp.Comp_Kernel(gp.sqexp,gp.sqexp,'kern_prod')]
t=0

fig,ax=plt.subplots(3,2,figsize=(20,30))

for i in kernels:
	if isinstance(i,gp.Comp_Kernel):
		param=compparams
		bound=compbounds
		a=gp.GP(i,Comp_Kernel=True)
		b=gp.GP(i,Comp_Kernel=True)
	else:
		a=gp.GP(i)
		b=gp.GP(i)
		param=params
		bound=bounds
	
	a.add_training(Xtrain,Ytrain,param,bound)
	f,mu,stdv=a.predict(Xpredict,False,n=1)
	print('a params ', a.par)

	b.add_training(Xtrain2,Ytrain2,param,bound)	
	f2,mu2,stdv2=b.predict(Xpredict,False,n=1)
	print('b params', b.par)
	


	ax[t,1].plot(Xtrain,Ytrain,'bs',ms=8)
	ax[t,1].plot(Xpredict,f)
	ax[t,1].fill_between(Xpredict.flat,mu-2*stdv,mu+2*stdv,color='#dddddd')
	ax[t,1].plot(Xpredict,mu,'k--',lw=2)
	ax[t,1].plot(Xpredict,Ypredict,'g--')
	ax[t,1].set_title('With training point in dip')
	ax[t,1].set_ylim([-3,3])

	ax[t,0].plot(Xtrain2,Ytrain2,'bs',ms=8)
	ax[t,0].plot(Xpredict,f2)
	ax[t,0].fill_between(Xpredict.flat,mu2-2*stdv2,mu2+2*stdv2,color='#dddddd')
	ax[t,0].plot(Xpredict,mu2,'k--',lw=2)
	ax[t,0].plot(Xpredict,Ypredict,'g--')
	ax[t,0].set_title('Without training point in dip')
	ax[t,0].set_ylim([-3,3])

	t += 1

plt.show()