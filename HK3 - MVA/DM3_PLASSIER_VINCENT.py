### PLASSIER Vincent : Master M2 MVA 2017/2018 - Optimisation Convexe - DM3


import numpy as np
import matplotlib.pyplot as plt


plt.ion()
plt.show()


## Q2: implementation of the barrier method
def funct(Q,p,A,b,t,v):
    y=t*np.dot(np.dot(v,Q),v)+np.dot(p,v)-np.sum(np.log(-np.dot(A,v)+b))
    return y

def grad(Q,p,A,b,t,v):
    c=(b-np.dot(A,v))
    gradient=t*(p+2*np.dot(Q,v))+np.sum(A/np.tensordot(c,np.ones(n),0),axis=0)
    return gradient


def Wolfe_search(Q,p,A,b,t,g,d,v):
    c1, c2 = .0001, 0.9
    h = 1 # initialisation
    y = funct(Q,p,A,b,t,v)
    a = c1*np.dot(d,g)
    b= c2*np.dot(d,g)
    h_g, h_d = 0, 0
    while funct(Q,p,A,b,t,v+h*d)>y+h*a or np.dot(d,grad(Q,p,A,b,t,v+h*d))<b:
        if funct(Q,p,A,b,t,v+h*d)>y+h*a:
            h_d = h
        elif np.dot(d,grad(Q,p,A,b,t,v+h*d))<b:
            h_g = h
        if h_d==0:
            h *= 1.1
        else:
            h = (h_g+h_d)/2
    return h


def Armijo_search(Q,p,A,b,t,g,d,v):
    y = funct(Q,p,A,b,t,v)
    h = 1 # initialisation
    a = .5*np.dot(d,g)
    while funct(Q,p,A,b,t,v+h*d)>y+h*a:
            h*=0.9
    return h


def centering_step(Q,p,A,b,t,v0,eps=.001,nit_max=1000,line_search=''):
    v_it=[] # liste des itérations
    n=Q.shape[0]
    err=1
    hess=np.zeros_like(A)
    nb_it_Newton = 0 # on initialise le compteur
    while err>eps and nb_it_Newton<nit_max:
        c=(b-np.dot(A,v0))
        g=grad(Q,p,A,b,t,v0)
        hess=np.dot(A.T,A/np.tensordot(c**2,np.ones(n),0))
        hess=2*t*Q+hess # matrice hessienne
        d=-np.linalg.solve(hess,g) # on calcule la direction de descente
        h=1
        if line_search=='Wolfe':
            h=Wolfe_search(Q,p,A,b,t,g,d,v0) 
        elif line_search=='Armijo':
            h=Armijo_search(Q,p,A,b,t,g,d,v0)
        v1=v0+h*d 
        err=np.linalg.norm(v1-v0) 
        v0=v1
        v_it.append(v0) # on rajoute le nouveau point à la liste
        nb_it_Newton+=1
    return v_it, nb_it_Newton


def barr_method(Q,p,A,b,v0,eps=.001,nit_max=50,mu=110,line_search=''):
    m=A.shape[0]
    v_it=[v0]
    t=1
    nb_it_Newton=0
    while m/t>=eps:
        v, nb_newton = centering_step(Q,p,A,b,t,v0,eps,line_search=line_search)
        v0=v[-1]
        v_it.append(v0)
        t=mu*t
        nb_it_Newton += nb_newton
    return v_it, nb_it_Newton


## Q3: Test
Lambda=10

n=np.random.randint(10,100)
X=np.random.randn(n,n)
y=np.random.randn(n)

Q=np.eye(n)/2
b=Lambda*np.ones(2*n)
p=-y
A=np.array([X,-X]).reshape(2*n,n)

f = lambda v : np.dot(v,np.dot(Q,v))+np.dot(p,v)

eps=0.001
v0=np.zeros(n)
v_it, nb_it_Newton = barr_method(Q,p,A,b,v0,eps,mu=20,line_search='Armijo') # None, Wolfe, Armijo
print('Nombre d utilisations de la méthode de Newton:', nb_it_Newton)

list_f_it = np.array([f(v) for v in v_it]) # pas très computationnel
f_star = np.min(list_f_it)

Err = list_f_it-f_star

plt.figure(1)
plt.clf()
plt.semilogx(Err)
plt.xlabel("itérations")
plt.ylabel("écart entre f(v) et f_star")
plt.title('Erreur en fonction de l itération')


## Nombre d'itérations dans barr_method en fonction de mu
list_mu = 4*np.arange(50)+2
list_long=[]

for mu in list_mu:
    v_it, nb_None = barr_method(Q,p,A,b,v0,eps,mu=mu)
    list_long.append(len(v_it)-1)
    
    
plt.figure(2)
plt.clf()
plt.plot(list_mu,list_long)
plt.title('Nombre d utilisation de Newton')


## Nombre d'itérations dans Newton en fonction de mu
list_mu = 4*np.arange(50)+2
list_nb_it_Newton = []

for mu in list_mu:
    v_it, nb_Wolfe = barr_method(Q,p,A,b,v0,eps,mu=mu,line_search='Wolfe')
    list_nb_it_Newton.append(nb_Wolfe)


list_nb_it_Newton = np.asarray(list_nb_it_Newton)

plt.figure(3)
plt.clf()
plt.plot(list_mu,list_nb_it_Newton, label='Wolfe')
plt.xlabel("mu")
plt.ylabel("itérations")
plt.title('Nombre d utilisation de Newton')

