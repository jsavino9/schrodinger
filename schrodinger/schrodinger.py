# coding: utf-8

import tensorflow as tf
import numpy as np
import math
np.set_printoptions(suppress=True)
tf.enable_eager_execution()

def integrate(fx,x):
    integral = 0
    for i in range(len(x)-1):
        integral += (x[i+1]-x[i])*(fx[i+1]+fx[i])/2
    return integral

def makearray(x,b):
    m = tf.Variable(tf.zeros((b,len(x))))
    for i in range(b):
        for j in range(len(x)):
            if i % 2 == 0:
                m = m[i,j].assign(math.cos(i//2*x[j]))
            else:
                m = m[i,j].assign(math.sin((i+1)//2*x[j]))
    return m

def v0hat(x,v,b):
    m = makearray(x,b)
    mt = tf.transpose(m)
    v = tf.constant(v, dtype = float)
    v0 = tf.Variable(tf.zeros(b, dtype=float))
    vhat = tf.reshape(tf.matmul([m],[mt]),(b,b))
    for i in range(b):
        v0sum = 0
        for j in range(len(x)):
           v0sum += m[i][j]*v[j]
        v0 = v0[i].assign(v0sum)
    v0 = tf.reshape(v0,(b,1))
    coef = tf.linalg.solve(vhat,v0)
    return coef

 
def hamiltonian(x,c,coeff,b):
    m = makearray(x,b)
    h = tf.Variable(tf.zeros((b,b), dtype=float))
    for i in range(b):
        for j in range(b):
            h = h[j,i].assign(c*m[i,j]*((i+1) // 2)**2 + m[i,j]*coeff[j][0])
    return h

def energies(h):
    e, cf = tf.linalg.eigh(h)
    emin = e[tf.argmin(e,0)]
    cfmin = cf[tf.argmin(e,0),:]
    return(emin,cfmin)

def main():
    vx = np.loadtxt("potential_energy.dat")
    x = vx[:,0]
    v = vx[:,1]
    c = 0
    b = 6
    coeff = v0hat(x,v,b)
    print(coeff)
    h = hamiltonian(x,c,coeff,b)
    print(h)
    emin,cfmin = energies(h)
    print('Min energy is {}' .format(emin))
    print('Coefficients for min energy wave function are {}' .format(cfmin))

main()

