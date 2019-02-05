## Plassier
import numpy as np
import scipy as sc
import numpy.random as rnd
import matplotlib.pyplot as plt
from numpy.linalg import norm
### Rosenbrock function and its gradient
def rosenbrock(x):
    y = np.asarray(x)
    return ((y[0] - 1)**2 + 100 * (y[1] - y[0]**2)**2)
def rosenbrock_grad(x):
    y = np.asarray(x)
    grad = np.zeros_like(y)
    grad[0] = 400 * y[0] * (y[0]**2 - y[1]) + 2 * (y[0] - 1)
    grad[1] = 200 * (y[1] - y[0]**2)
    return grad
##Q1
xmin,xmax,ymin,ymax=-1.5,1.5,-0.5,1.5
xx=np.linspace(xmin,xmax,500)
yy=np.linspace(ymin,ymax,500)
[X,Y]=np.meshgrid(xx,yy)
fXY=rosenbrock([X,Y])

plt.figure(1)
plt.clf()
CS=plt.contour(X,Y,fXY,np.logspace(-0.5,3.5,20,base=10))
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Lignes de niveau de Rosenbrock')
plt.show()
##Q2
epsilon=10**-8
def GPF(h,x0,f_grad,epsilon=10**-8,n_max=1000):
    error=1
    increment=0
    XX=[x0]
    while error > epsilon and increment < n_max :
        x1=x0-h*f_grad(x0)
        error=np.linalg.norm(x0-x1,2)
        x0=x1
        increment+=1
        XX.append(x1)
    return np.array(XX)
##Q3
h=10**-3
x0=np.array([-1,1])
XX=GPF(h,x0,rosenbrock_grad,n_max=10000)

plt.figure(2)
plt.clf()
plt.plot(XX[:,0],XX[:,1],'ro')
CS=plt.contour(X,Y,fXY,np.logspace(-0.5,3.5,20,base=10))
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Méthode de descente par gradient pour h=5*0.01')
plt.show()
##Q4
h=5*0.01
epsilon=10**-8
def GPFn(h,x0,f_grad,epsilon=10**-8,n_max=10000):
    error=1
    increment=0
    x0=np.asarray(x0)
    XX=[x0]
    grad=f_grad(x0)
    while increment < n_max and error>epsilon:
        x1=x0-h*grad/norm(grad,2)
        grad=f_grad(x1)
        error=norm(x1-x0)
        x0=x1
        increment+=1
        XX.append(x1)
    return np.array(XX)
XX=GPFn(h,x0,rosenbrock_grad,epsilon)
plt.figure(3)
plt.clf()
plt.plot(XX[:, 0],XX[:, 1],color='r',markersize=0.6)
CS=plt.contour(X,Y,fXY,np.logspace(-0.5,3.5,20,base=10))
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Méthode de descente normalisé pour h=5*0.01')
plt.show()
## Problème quadratique
def mk_quad(m, M, ndim=2):
    def quad(x):
        y = np.copy(np.asarray(x))
        y = y**2
        y[0]=y[0]*M
        y[1]=y[1]*m
        return np.sum(y,axis=0)
    def quad_grad(x):
        y = np.asarray(x)
        scal = np.ones(ndim)
        scal[0] = M
        scal[1] = m
        return 2 * scal * y
    return quad, quad_grad
##Q1 
m,M=1,3 # On les choisit
xmin,xmax,ymin,ymax=-1.5,1.5,-1.5,1.5
xx=np.linspace(xmin,xmax,500)
yy=np.linspace(ymin,ymax,500)
[X,Y]=np.meshgrid(xx,yy)
fXY=mk_quad(m,M,ndim=2)[0]([X,Y])

plt.figure(4)
plt.clf()
CS = plt.contour(X, Y, fXY,np.logspace(-0.5,3.5,20,base=10),colors='darkslateblue')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Lignes de niveaux lorque m=%s et M=%s'%(m,M))
plt.axis([-1.5,1.5,-1.5,1.5])
plt.show()
##Q2
x0=np.array([-1,1])
def methode_gradient_optimal(x0,m,M,epsilon=10**-8,n_max=10**3):
    increment=0
    error=1
    XX=[x0]
    while error>epsilon and increment<n_max:
        d=mk_quad(m, M, ndim=2)[1](x0)
        x1=x0-d/(m+M)
        x0=x1
        XX.append(x0)
        increment+=1
    return np.array(XX)

XX=methode_gradient_optimal(x0,m,M,epsilon=10**-8,n_max=10**3)
plt.figure(5)
plt.clf()
plt.plot(XX[:, 0],XX[:, 1],'ro')
CS = plt.contour(X, Y, fXY,np.logspace(-0.5,3.5,20,base=10),colors='darkslateblue')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Lignes de niveaux lorque m=%s et M=%s'%(m,M))
plt.axis([-1.5,1.5,-1.5,1.5])
plt.show()
##Q3
m,M=1,3
H=np.linspace(0.001,1/(m+M)+0.1,300)

f_grad=mk_quad(m, M, ndim=2)[1]
Liste=[np.log(norm(GPF(h,x0,f_grad,epsilon=10**-16,n_max=2000),axis=1)) for h in H]
Liste2=[Liste[i][1:]/Liste[i][:-1]  for i in range(len(Liste))]
Ordre=[np.mean(Liste2[i][len(Liste2[i])/2:]) for i in range(len(Liste))]

plt.figure(6)
plt.clf()
plt.plot(H,Ordre,color='darkslateblue')
plt.xlabel('h')
plt.ylabel('Ordre de convergence')
plt.title('Ordre de convergence en fonction de h lorque m=%s et M=%s'%(m,M))
plt.show()
##Q4
h,h_opt=10**-2,1/(m+M)
m,M=1,3
dimension=[2,3,5,10,20,30,40,50]

Erreur=[]
Nb_iteration=[]
for dim in dimension:
    x0=np.ones(dim)
    Erreur.append(norm(GPF(h,x0,mk_quad(m, M, dim)[1],epsilon=0,n_max=1000)))
    Nb_iteration.append(len(GPF(h_opt,x0,mk_quad(m, M, dim)[1],epsilon=10**-8,n_max=10**10)))
Erreur=np.array(Erreur)
print('Les ratios des erreurs:',Erreur[1:]/Erreur[0])
print('Le nombre d itérations pour h_opt:',Nb_iteration)
plt.figure(7)
plt.clf()
plt.subplot(2,1,1)
plt.scatter(dimension[1:],Erreur[1:]/Erreur[0],marker = '+',color = 'green',linewidth=12,label='Ratios des erreurs pour h')
plt.axis([0,60,1,8])
plt.legend()
plt.subplot(2,1,2)
plt.scatter(dimension,Nb_iteration,marker = '+',color = 'red',linewidth=12,label='Nombre d itérations pour h_opt')
plt.axis([0,60,29,32])
plt.legend()
plt.show()
### Partie 2
def Armijo(x0,f,f_grad,c,L,epsilon,n_max):
    increment=0
    error=1
    XX=[x0]
    while error>epsilon and increment<n_max:
        d=-f_grad(x0) # direction de descente
        h=1/L # initialise
        y=f(x0)
        A=c*norm(d)**2
        while f(x0+h*d)>y-h*A:
            h*=0.9
        x1=x0+h*d
        error=norm(x1-x0)
        x0=x1
        XX.append(x0)
        increment+=1
    return np.array(XX)                

def Wolfe(x0,f,f_grad,c1,c2,L,alpha,epsilon,n_max):
    increment=0
    error=1
    XX=[x0]
    while error>epsilon and increment<n_max:
        d=-f_grad(x0) # direction de descente
        h=1/L # initialise
        y=f(x0)
        A=c1*norm(d)**2
        h_g,h_d=0,0
        while f(x0+h*d)>y-h*A or np.dot(d,f_grad(x0+h*d))<-c2*norm(d)**2:
            if f(x0+h*d)>y-h*A:
                h_d=h
            elif np.dot(d,f_grad(x0+h*d))<-c2*norm(d)**2:
                h_g=h
            if h_d==0:
                h*=alpha
            else:
                h=(h_g+h_d)/2
        x1=x0+h*d
        error=norm(x1-x0)
        x0=x1
        XX.append(x0)
        increment+=1
    return np.array(XX)
##Q2
x0=np.array([-1,1])
n_max,espsilon=2000,10**-16
h=10**-3
m,M=1,5
alpha=1.1
c,c1,c2,L=.7,.5,.9,100

f1,f1_grad=rosenbrock,rosenbrock_grad
x_min=np.array([1,1]) # pour Rosenbrock
# x_min=scipy.optimize.fmin(f,x0) # Cas général
XX_grad = GPF(h,x0,f1_grad,epsilon,n_max)
XX_Armijo = Armijo(x0,f1,f1_grad,c,L,epsilon,n_max)
XX_Wolfe = Wolfe(x0,f1,f1_grad,c1,c2,L,alpha,epsilon,n_max)
YY_g=norm(XX_grad-x_min,axis=1)
YY_A=norm(XX_Armijo-x_min,axis=1)
YY_W=norm(XX_Wolfe-x_min,axis=1)

temps_grad=np.linspace(0,len(YY_g)-1,len(YY_g))
temps_Armijo=np.linspace(0,len(YY_A)-1,len(YY_A))
temps_Wolfe=np.linspace(0,len(YY_W)-1,len(YY_W))

plt.figure(8)
plt.clf()
plt.axis([0,n_max,0,norm(x0-x_min)])
plt.plot(temps_grad,YY_g, label='gradient')
plt.plot(temps_Armijo,YY_A, label='Armijo')
plt.plot(temps_Wolfe,YY_W,label='Wolfe')
plt.xlabel("n-ième itération")
plt.ylabel("distance au minimum")
plt.title('c=%s,c1=%s,c2=%s et h=%s,'%(c,c1,c2,h))
plt.show()
## f(x)-f(x_min):
ZZ_g=abs(f1(XX_grad.T)-f1(x_min))
ZZ_A=abs(f1(XX_Armijo.T)-f1(x_min))
ZZ_W=abs(f1(XX_Wolfe.T)-f1(x_min))

plt.figure(9)
plt.clf()
plt.axis([0,n_max,0,norm(f1(x0)-f1(x_min))])
plt.plot(temps_grad,ZZ_g, label='gradient')
plt.plot(temps_Armijo,ZZ_A, label='Armijo')
plt.plot(temps_Wolfe,ZZ_W,label='Wolfe')
plt.xlabel("n-ième itération")
plt.ylabel("distance au minimum")
plt.title('c=%s,c1=%s,c2=%s et h=%s,'%(c,c1,c2,h))
plt.show()
##Q3
m,M=1,5
epsilon=10**-16
f2,f2_grad=mk_quad(m, M, ndim=2)[0],mk_quad(m, M, ndim=2)[1]
x_min=np.array([0,0])

XX_grad = GPF(h,x0,f2_grad,epsilon,n_max)
XX_Armijo = Armijo(x0,f2,f2_grad,c,L,epsilon,n_max)
XX_Wolfe = Wolfe(x0,f2,f2_grad,c1,c2,L,alpha,epsilon,n_max)
YY_g=norm(XX_grad-x_min,axis=1)
YY_A=norm(XX_Armijo-x_min,axis=1)
YY_W=norm(XX_Wolfe-x_min,axis=1)

temps_grad=np.linspace(0,len(YY_g)-1,len(YY_g))
temps_Armijo=np.linspace(0,len(YY_A)-1,len(YY_A))
temps_Wolfe=np.linspace(0,len(YY_W)-1,len(YY_W))

plt.figure(10)
plt.clf()
plt.axis([0,n_max,0,norm(x0-x_min)])
plt.plot(temps_grad,YY_g, label='gradient')
plt.plot(temps_Armijo,YY_A, label='Armijo')
plt.plot(temps_Wolfe,YY_W,label='Wolfe')
plt.xlabel("n-ième itération")
plt.ylabel("distance au minimum")
plt.title('c=%s,c1=%s,c2=%s et h=%s,'%(c,c1,c2,h))
plt.show()
## f(x)-f(x_min):
ZZ_g=abs(f2(XX_grad.T)-f2(x_min))
ZZ_A=abs(f2(XX_Armijo.T)-f2(x_min))
ZZ_W=abs(f2(XX_Wolfe.T)-f2(x_min))

plt.figure(11)
plt.clf()
plt.axis([0,n_max,0,norm(f2(x0)-f2(x_min))])
plt.plot(temps_grad,ZZ_g, label='gradient')
plt.plot(temps_Armijo,ZZ_A, label='Armijo')
plt.plot(temps_Wolfe,ZZ_W,label='Wolfe')
plt.xlabel("n-ième itération")
plt.ylabel("distance au minimum")
plt.title('c=%s,c1=%s,c2=%s et h=%s,'%(c,c1,c2,h))
plt.show()
### Gradient conjugué
def Polak_Ribiere(x0,f,f_grad,epsilon=10**-8,n_max=10**4):
    c1=0.01
    c2=0.9
    error=1
    increment=0
    XX=[x0]
    x1=x0
    d=0
    g0=f_grad(x0)
    while error>epsilon and increment<n_max:
        g1=f_grad(x1)
        x0=x1
        beta=np.dot((g1-g0),g1)/norm(g0)**2
        d=-g1+beta*d
        d=-np.sign(np.sum(g1*d))*d # bien direction de descente
        h=0.01 # initialise
        y=f(x0)
        A=np.dot(d,g1)
        h_g,h_d=0,0
        i=0
        while (f(x0+h*d)>y+h*c1*A or np.dot(d,f_grad(x0+h*d))<c2*A) and i<10**5:
            i+=1
            if f(x0+h*d)>y+h*c1*A:
                h_d=h
            elif np.sum(d*f_grad(x0+h*d))<c2*A:
                h_g=h
            if h_d==0:
                h*=1.1
            else:
                h=(h_g+h_d)/2
        x1=x0+h*d
        g0=g1
        error=norm(x1-x0)
        increment+=1
        XX.append(x1)
    return np.array(XX)

n_max=10**5
epsilon=10**-8
x0=np.array([-1,1])

f1,f1_grad=rosenbrock,rosenbrock_grad
l_PR=len(Polak_Ribiere(x0,f1,f1_grad,epsilon,n_max))
l_g=len(GPF(h,x0,f1_grad,epsilon,n_max))
l_A=len(Armijo(x0,f1,f1_grad,c,L,epsilon,n_max))
l_W=len(Wolfe(x0,f1,f1_grad,c1,c2,L,alpha,epsilon,n_max))
print('Nombre itéations pour epsilon=%s avec Rosenbrock: \n Polak_Ribiere=%s'%(epsilon,l_PR))
print('Gradient=%s \n Armijo=%s \n Wolfe=%s'%(l_g,l_A,l_W))

f2,f2_grad=mk_quad(m, M, ndim=2)[0],mk_quad(m, M, ndim=2)[1]
l_PR=len(Polak_Ribiere(x0,f2,f2_grad,epsilon,n_max))
l_g=len(GPF(h,x0,f2_grad,epsilon,n_max))
l_A=len(Armijo(x0,f2,f2_grad,c,L,epsilon,n_max))
l_W=len(Wolfe(x0,f2,f2_grad,c1,c2,L,alpha,epsilon,n_max))
print('Nombre itéations pour epsilon=%s cas quadratique: \n Polak_Ribiere=%s'%(epsilon,l_PR))
print('Gradient=%s \n Armijo=%s \n Wolfe=%s'%(l_g,l_A,l_W))
## Cas quadratique
def grad_conj_pas_op(x0,f,f_grad,ndim = 2):
    XX=[x0]
    x1=x0
    d=np.zeros(ndim)
    while True:
        g0=f_grad(x0)
        g1=f_grad(x1)
        if norm(g1)<10**-20:
            break
        x0=x1
        c=norm(g1)**2/norm(g0)**2
        d=-g1+c*d
        d=-np.sign(np.sum(g1*d))*d # bien direction de descente
        h=-norm(g1)**2/np.dot(g1,f_grad(d))
        x1=x0+h*d
        XX.append(x1)
    return np.array(XX)

m,M=1,3
dimension=np.linspace(2,100,dtype = int)
Erreur=[]
Vs=[]
for d in dimension:
    x0=np.ones(d)
    f2,f2_grad=mk_quad(m, M, ndim=d)[0],mk_quad(m, M, ndim=d)[1]
    A,b=f2_grad(np.eye(d))/2,np.zeros(d)
    x = grad_conj_pas_op(x0,f2,f2_grad,ndim = d)
    Erreur.append(len(x))
    L=[]
    number=lambda x: L.append(x)
    sc.sparse.linalg.cg(A, b, x0, tol=espsilon,callback=number)[0]
    Vs.append(len(L))
Erreur=np.array(Erreur)
plt.figure(12)
plt.clf()
plt.subplot(2,1,1)
plt.axis([4,101,0,8])
plt.scatter(dimension,Erreur,marker = '+',color = 'green')
plt.title('Notre algorithme')
plt.subplot(2,1,2)
plt.scatter(dimension,Vs,marker = '+',color = 'red')
plt.title('Algorithme python')
plt.axis([4,101,0,8])
plt.show()