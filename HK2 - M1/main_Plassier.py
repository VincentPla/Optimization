### PLASSIER : Minimisation par processus gaussien

## Acquisition comprimée
import numpy as np
import matplotlib.pyplot as plt
from tp2_tools import *
import warnings
warnings.filterwarnings('ignore')
from ista_Casabianca_Plassier import ista

##Q2
lambda_l1 = 1
y, A = noisy_observations()

##Q5
f = lambda x : np.sum((y-np.dot(A,x))**2)
g = lambda x: np.sum(abs(x))
fun_total = lambda x: f(x)+lambda_l1*g(x)
L=2*np.linalg.norm(np.dot(A.T,A),2) # f est L-lipschitzienne
grad_f = lambda x : 2*np.array([np.dot(np.dot(A.T,A),x)-np.dot(A.T,y),L])
h = lambda x,gamma : (x-np.sign(x)*gamma)*(abs(x)-gamma>=0)
prox_op_g = lambda x,gamma : np.array(h(x,gamma))

##Q6
n_it=1000
dim=np.shape(A)[1]
I=ista(dim,prox_op_g, grad_f, fun_total, lambda_l1, n_it)

plot_image(I[0])
plt.legend()
plt.show()

##Q7
H=np.arange(0,n_it+1,1)
Liste=np.array(I[1])*H

p=np.polyfit(H[int(n_it/1.5):],Liste[int(n_it/1.5):],1) # a*x+b

plt.figure(7)
plt.clf()
plt.plot(H,Liste-p[0]*H,label='polynome') # F(x_k)-k*F(x_*)
plt.xlabel('k-ième itération')
plt.ylabel('k*F(x_k)-a*k-b')
plt.title('k*F(x_k)-a*k-b en fonction de k')
plt.legend()
plt.show()

##Q8
Lambda=10*0.1**np.linspace(0,5,15)

for lambd in Lambda:
    I8=ista(dim,prox_op_g, grad_f, fun_total, lambd, n_it=300)
    plot_image(I8[0])
    plt.title('Image pour lambda=%s' %(lambd))
    plt.show()