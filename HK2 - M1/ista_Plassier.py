### PLASSIER : Minimisation par precessus gaussien

## Implementation de ISTA
import numpy as np
import matplotlib.pyplot as plt

##Q2
def ista(dim,prox_op_g, grad_f, fun_total, lambda_l, n_it=100):
    x0=np.zeros(dim)
    gamma=1/grad_f(x0)[1]
    fun_iterate=[fun_total(x0)]
    for k in range(n_it):
        x1=prox_op_g(x0-gamma*grad_f(x0)[0],lambda_l*gamma)
        x0=x1
        fun_iterate.append(fun_total(x0))
    return x0,fun_iterate

## Récursive : un tout petit peu plus efficace
def ista_re(dim,prox_op_g, grad_f, fun_total, lambda_l, n_it=100):
    x0=np.zeros(dim)
    gamma=1/grad_f(x0)[1]
    fun_iterate=[fun_total(x0)]
    def recursif(x0):
        x1=prox_op_g(x0-gamma*grad_f(x0)[0],lambda_l*gamma)
        fun_iterate.append(fun_total(x1))
        if len(fun_iterate)==101:
            return x1,fun_iterate
        return recursif(x1)
    XX=recursif(x0)
    return XX