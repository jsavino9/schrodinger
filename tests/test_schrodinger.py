#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `schrodinger` package."""

import pytest
import numpy as np
import tensorflow as tf
import math
from schrodinger import schrodinger
from schrodinger import cli
tf.enable_eager_execution()


@pytest.fixture

def test_start():
	'''
	empty space
	'''
def test_makearray():
	x = [0,math.pi/2,math.pi]
	b = 3
	m = schrodinger.makearray(x,b)
	assert int(m[0][0]) == int(math.cos(0))
	assert int(m[1][0]) == int(math.sin(0))
	assert int(m[1][1]) == int(math.sin(math.pi/2))

def test_coeffs():
	x = [0,math.pi/2,math.pi]
	b = 3
	v = [0,6,0]
	coefs = schrodinger.v0hat(x,v,b)
	assert int(coefs[0]) == int(0)
	assert int(coefs[1]) == int(6)
	assert int(coefs[2]) == int(0)

def test_h():
	x = [0,math.pi/2,math.pi]
	b = 3
	v = [0,6,0]
	c = 0
	coefs = schrodinger.v0hat(x,v,b)
	h = schrodinger.hamiltonian(x,c,coefs,b)
	assert int(h[0][0]) == int(0)
	assert int(h[1][0]) == int(6)

def test_e():
	x = [0,math.pi/2,math.pi]
	b = 3
	v = [0,6,0]
	c = 0
	coefs = schrodinger.v0hat(x,v,b)
	h = schrodinger.hamiltonian(x,c,coefs,b)
	emin,cfmin = schrodinger.energies(h)
	assert(float(emin)) == float(-3.708204746246338)
