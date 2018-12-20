===========
1D Schrodinger
===========


.. image:: https://img.shields.io/pypi/v/schrodinger.svg
        :target: https://pypi.python.org/pypi/schrodinger

.. image:: https://img.shields.io/travis/jsavino9/schrodinger.svg
        :target: https://travis-ci.org/jsavino9/schrodinger

.. image:: https://coveralls.io/repos/github/jsavino9/schrodinger/badge.svg?branch=master
:target: https://coveralls.io/github/jsavino9/schrodinger?branch=master



This application takes in a list of data detailing potential energy as a function of x, the size of a basis set, and the scaling factor for kinetic energy and applies this to the 1D Schrodinger equation.  It reports the minimum energy of the system and the wavefunction corresponding to that lowest energy state.

* Free software: MIT license
=========================
Installation Instructions
=========================

This program is intended to be run on Python 3.

Required packages: numpy, tensorflow

*packages can be installed using "pip install [package]"*

After downloading the required packages, go to the repository https://github.com/jsavino9/schrodinger.  Click clone/download, and click "download as zip".  Once the file is downloaded, it can be unzipped.  An alternative option is to clone the repository.

===================
Running the Program
===================

Navigate to the source directory at ../schrodinger

Run the program by typing "python schrodinger/schrodinger.py".  If running through a python interpreter, just typing "schrodinger/schrodinger.py" will suffice.

There are two files that are required to run the program.  The first is the potential energy file.  An example file can be found in the repository.  The first column holds the x values and the second holds the potential energies.  The number of entries in both must be equal.  The second is the params file.  The first entry is c, the scaling factor for the kinetic energy, and the second is b, the size of the basis set.  b must be an integer.

Note: The if the size of the basis set contains more values than the domain/potential energy, then it may result in unpredictable behavior. 

Outputs
-------

emin: this is the minimum value of the energy for this system

cfmin: these are the coefficients belonging to the wave function expressed in the desired basis set, such that a0 + a1*sin(x) + b1*cos(x) + a2*sin(2x) + ...


Credits
-------
Author: James Savino

Project was completed for CHE 477 at the University of Rochester under Professor Andrew White

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
