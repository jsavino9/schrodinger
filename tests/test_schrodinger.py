#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger` package."""

import pytest
import numpy as np
import tensorflow as tf
import math
from schrodinger import schrodinger
tf.enable_eager_execution()


@pytest.fixture

def test_start():
	'''
	empty space
	'''
def test_makearray():
	'''
	Tests the makearray function
	'''
	x = [0,math.pi/2,math.pi] #set x
	b = 4 #set b
	m = schrodinger.makearray(x,b) #set m
	#test against known values
	assert int(m[0][0]) == int(math.cos(0))
	assert int(m[1][0]) == int(math.sin(0))
	assert int(m[1][1]) == int(math.sin(math.pi/2))
	assert int(m[0][3]) == int(0)

def test_coeffs():
	'''
	Makes sure the coefficients are calculated correctly
	'''
	#set parameters
	x = [0,math.pi/2,math.pi]
	b = 3
	v = [0,6,0]
	#calculate coefficients and compare to known values
	coefs = schrodinger.v0hat(x,v,b)
	assert int(coefs[0]) == int(0)
	assert int(coefs[1]) == int(6)
	assert int(coefs[2]) == int(0)

def test_h():
	'''
	Makes sure the hamiltonian is constructed properly
	'''
	#set parameters
	x = [0,math.pi/2,math.pi]
	b = 3
	v = [0,6,0]
	c = 0
	coefs = schrodinger.v0hat(x,v,b)
	#calculate h and compare to known values
	h = schrodinger.hamiltonian(x,c,coefs,b)
	assert int(h[0][0]) == int(0)
	assert int(h[1][0]) == int(6)

def test_e():
	'''
	Tests the energies function
	'''
	#set parameters
	x = [0,math.pi/2,math.pi]
	b = 3
	v = [0,6,0]
	c = 0
	coefs = schrodinger.v0hat(x,v,b)
	h = schrodinger.hamiltonian(x,c,coefs,b)
	#calculate emin and cfmin and compare to known values
	emin,cfmin = schrodinger.energies(h)
	assert(float(emin)) == float(-3.708204746246338)

def test_fileopen():
	'''
	Tests to see if files open properly
	'''
	#Open files
	x,v,c,b = schrodinger.read_data()
	assert type(b) == int #makes sure b is int
	assert type(c) == float #makes sure c is float
	assert len(x) == len(v) #makes sure the number of entries in potential energy equals domain
