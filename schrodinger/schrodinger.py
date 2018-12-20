# coding: utf-8

import tensorflow as tf
import numpy as np
import math
np.set_printoptions(suppress=True)
tf.enable_eager_execution()

def makearray(x,b):
	'''
	Makes 2D array using the domain of the potential energy function
	and the basis sets

	arguments:
	x: domain of the potential energy function (array)
	b: the size of the basis set (int)

	returns:
	m: the 2D array/matrix (2d array)
	'''
	if b > len(x):
		m = tf.Variable(tf.zeros((b,b)))
	else:
		m = tf.Variable(tf.zeros((b,len(x))))
	#Assign value corresponding to sine or cosine depending on index
	for i in range(b):
		for j in range(len(x)):
			if i % 2 == 0:
				m = m[i,j].assign(math.cos(i//2*x[j]))
			else:
				m = m[i,j].assign(math.sin((i+1)//2*x[j]))
	return m

def v0hat(x,v,b):
	'''
	Calculates the coefficients for v0hat 

	arguments:
	x: domain of the potential energy function (array)
	v: the potential energy function (array)
	b: the size of the basis set (int)

	returns:
	coef: the coefficients (array)
	'''

	m = makearray(x,b) #makes array for the basis set
	mt = tf.transpose(m) #gets transpose of basis array
	v = tf.constant(v, dtype = float) #Makes v a tf.constant so it works in tensorflow
	v0 = tf.Variable(tf.zeros(b, dtype=float)) #Initialize v0
	vhat = tf.reshape(tf.matmul([m],[mt]),(b,b)) #vhat is m * transpose of m
	#Calculates v0 by multiplying the basis array by the potential energy function
	for i in range(b):
		v0sum = 0
		for j in range(len(x)):
			v0sum += m[i][j]*v[j]
		v0 = v0[i].assign(v0sum)
	v0 = tf.reshape(v0,(b,1)) #Reshape v0 so it can be used in linalg.solve
	coef = tf.linalg.solve(vhat,v0) #Get the coefficients by solving vhat-1*b = v0
	return coef

 
def hamiltonian(x,c,coeff,b):
	'''
	Creates the hamiltonian by adding the kinetic and potential energy terms

	arguments:
	x: domain of the potential energy function (array)
	c: scaling constant for the kinetic energy (float)
	coeff: coefficients for v0hat (array)
	b: size of the basis set (int)

	returns:
	h: the hamiltonian (array)
	'''

	m = makearray(x,b) #make the basis set array
	h = tf.Variable(tf.zeros((b,b), dtype=float)) #initialize h, which is a bxb matrix
	#computes h: h = k + u, where k = c*del^2*psi(x) and u = v0(x)*psi(x)
	for i in range(b):
		for j in range(b):
			h = h[j,i].assign(c*m[i,j]*((i+1) // 2)**2 + m[i,j]*coeff[j][0])
	return h

def energies(h):
	'''
	Calculates the minimum energy and corresponding coefficients of the hamiltonian

	arguments: 
	h: the hamiltonian (array)

	returns:
	emin: minimum energy (float)
	cfmin: coefficients of lowest energy wavefunction (array)
	'''
	e, cf = tf.linalg.eigh(h) #get eigevalues and vectors to find e and coefficients
	emin = e[tf.argmin(e,0)] #get emin
	cfmin = cf[tf.argmin(e,0),:] #get cfmin from index of emin
	return(emin,cfmin)

def read_data():
	'''
	Read data files to get x, v, c, b
	
	inputs:
	potential_energy.dat: This file contains two columns, where first contains x values and second contains v values
	params.dat: This file has two single-item rows: first is c, second is b

	returns:
	x: the x values (array)
	v: the v values (array0
	c: scaling constant for kinetic energy (float)
	b: size of basis set (int)
	'''
	#Load data files
	vx = np.loadtxt("schrodinger/potential_energy.dat")
	param = np.loadtxt("schrodinger/params.dat")
	#x and v
	x = vx[:,0]
	v = vx[:,1]
	#c and b
	c = float(param[0])
	b = int(param[1])
	return x,v,c,b

def main():
	'''
	Finds the lowest energy and corresponding wave function.
	'''
	#Get data
	x,v,c,b = read_data()
	#find the v0hat coefficients
	coeff = v0hat(x,v,b)
	#find the hamiltonian
	h = hamiltonian(x,c,coeff,b)
	#get emin and cfmin
	emin,cfmin = energies(h)
	#print results
	print('Min energy is {}' .format(emin))
	print('Coefficients for corresponding wave function are {}' .format(cfmin))

main()

