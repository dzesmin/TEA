#! /usr/bin/env python

# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
# 
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Lead scientist and coder Jasmina Blecic, 
# assistant coder for the first pre-release Oliver M. Bowman.  
#
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# We welcome your feedback, but do not guarantee support.  Please send
# feedback or inquiries to both:
# 
# Jasmina Blecic <jasmina@physics.ucf.edu>
# Joseph Harrington <jh@physics.ucf.edu>
# 
# or alternatively,
# 
# Jasmina Blecic and Joseph Harrington
# UCF PSB 441
# 4000 Central Florida Blvd
# Orlando, FL 32816-2385
# USA
# 
# Thank you for testing TEA!
# ******************************* END LICENSE *******************************

from TEA_config import *

import os
import numpy as np
from sys import argv
from sympy.core    import Symbol
from sympy.solvers import solve

import format as form

# =============================================================================
# This code produces an initial guess for the first iteration of TEA by 
# fulfilling the mass balance condition, sum_i(ai_j * y_i) = bj (equation (16) 
# in TEA Document), where i is species index, j is element index, a's
# are stoichiometric coefficients, and b's are elemental fractions by number, 
# i.e., ratio of number densities of element 'j' to the total number densities 
# of all elements in the system (see end of the Section 1.1 of the TEA
# document). The code writes the result into machine- and human-readable files. 
#
# It begins by making a directory for the output results, reading the header
# file and importing all relevant data. To satisfy the mass balance equation
# some yi variables must remain as free parameters. The number of free 
# parameters is set to the number of total elements in the system, thus 
# ensuring that the mass balance equation can be solved for any number of input
# elements and output species the user chooses. The code locates a chunk 
# of species (y_i) containing a sum of ai_j values that forbids ignoring any
# element in the system (sum of the ai_j values in a column must not be zero). 
# This condition is necessary as the mass balance equation can not be solved
# for all elements otherwise. This chunk is used as a set of free variables
# in the system. The initial scale for other y_i variables are set to a known,
# arbitrary number. Initially, a the starting values for the known species
# are set to 0.1 moles, and the mass balance equation is calculated. If that 
# does not produce all positive mole numbers, the code automatically sets known
# parameters to 10 times smaller and tries again. Actual mole numbers for the
# initial guesses of y_i are arbitrary, as TEA only requires a balanced 
# starting point to initialize minimization. After all iterations are run, the
# solution will converge regardless of the input used. The goal of this code is
# to find a positive set of non-zero mole numbers to satisfy this requirement. 
# Finally, the code calculates y_bar, initializes the iteration number, delta,
# and delta_bar to zero and writes results into machine- and human-readable
# output files. 
#
# This code is called by runatm.py and runsingle.py.
# =============================================================================

# Read run-time arguments
header = argv[1:][0]              # Name of header file
desc   = argv[1:][1]              # Description of the run

# Create and name outputs and results directories if they do not exist
datadir   = 'outputs/' + 'transient/' + desc # location of storage directory

# If output directory does not exist already, make it
if not os.path.exists(datadir): os.makedirs(datadir)

# Read in values from header file
pressure, temp, i, j, speclist, a, b, c = form.readheader(header)

# Print b values for debugging purposes
if doprint:
    print("b values: " + str(b))

# Find chunk of ai_j array that will allow the corresponding yi values
#      to be free variables such that all elements are considered
for n in np.arange(i - j + 1):
    # Get lower and upper indices for chunk of ai_j array to check
    lower = n
    upper = n + j
    
    # Retrieve chunk of ai_j that would contain free variables
    a_chunk = a[lower:upper]
    
    # Sum columns to get total of ai_j in chunk for each species 'j'
    check = map(sum,zip(*a_chunk))
    
    # Look for zeros in check. If a zero is found, this chunk of data can't 
    # be used for free variables, as this signifies an element is ignored
    has_zero = 0 in check
    
    # If zero not found, create list of free variables' indices
    if has_zero == False:
        free_id = []
        for m in np.arange(j):
            if doprint == True:
                print('Using y_' + np.str(n + m + 1) + ' as a free variable')
            free_id = np.append(free_id, n + m)
        break

# Set initial guess of non-free y_i 
scale = 0.1

# Assume that all or some y_i are negative or zeros
nofit = True

# Loop until all y_i are non-zero positive
while nofit:
    # Set up list of 'known' initial mole numbers before and after free chunk
    pre_free = np.zeros(free_id[0]) + scale
    post_free = np.zeros(i - free_id[-1] - 1) + scale
    
    # Set up list of free variables
    free = []
    for m in np.arange(j):
        name = 'y_unknown_' + np.str(m)
        free = np.append(free, Symbol(name))

    # Combine free and 'known' to make array of y_initial mole numbers
    y_init = np.append(pre_free,      free)
    y_init = np.append(  y_init, post_free)

    # Make 'j' equations satisfying mass balance equation (16) in TEA Document:
    # sum_i(ai_j * y_i) = b_j
    eq = [[]]
    for m in np.arange(j):
        rhs = 0
        for n in np.arange(i):
            rhs += a[n, m] * y_init[n]
        rhs -= b[m]
        eq = np.append(eq, rhs)
        
    # Solve system of linear equations to get free y_i variables
    result = solve(list(eq), list(free))
    
    # Correct for no-solution-found results. 
    # If no solution found, decrease scale size.
    if result == []:
        scale /= 10
        if doprint:
            print("Correcting initial guesses for realistic mass. \
                    Trying " + str(scale) + "...")

    # Correct for negative-mass results.  If found, decrease scale size.
    else:
        # Assume no negatives and check
        hasneg = False    
        for m in np.arange(j):
            if result[free[m]] < 0: 
                hasneg = True
        # If negatives found, decrease scale size
        if hasneg:
            scale /= 10
            if doprint:
                print("Negative numbers found in fit.")
                print("Correcting initial guesses for realistic mass. \
                        Trying " + str(scale) + "...")
        # If no negatives found, exit the loop (good fit is found)
        else:
            nofit = False
            if doprint:
                print(str(scale) + " provided a viable initial guess.")
    
# Gather the results
fit = []
for m in np.arange(j):
    fit = np.append(fit, result[free[m]])

# Put the result into the final y_init array
y_init[free_id[0]:free_id[j-1]+1] = fit

# This part of the code is only for debugging purposes 
# It rounds the values and checks whether the balance equation is satisfied
# No values are changed and this serves solely as a check 
if doprint == True:
    print('\nCHECKS:')
for m in np.arange(j):
    bool = round((sum(a[:,m] * y_init[:])), 2) == round(b[m], 2)
    if bool == True:
        if doprint == True:
            print('Equation ' + np.str(m+1) + ' is satisfied.')
    if bool == False:
        print('Equation ' + np.str(m+1) + \
                         ' is NOT satisfied. Check for errors!')

# Set iteration number to zero
it_num    = 0

# Put all initial mole numbers in y array
y         = y_init

# Make y_bar (sum of all y values)
y_bar     = np.sum(y)

# Initialize delta variables to 0. (this signifies the first iteration)
delta     = np.zeros(i)   
delta_bar = np.sum(delta) 

# Name output files with corresponding iteration number name
file       = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                              '-machine-read.txt'
file_fancy = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                                    '-visual.txt'

# Put results into machine readable file
form.output(datadir, header, it_num, speclist, y, y, delta, \
            y_bar, y_bar, delta_bar, file, doprint)

# Put results into human readable file
form.fancyout(datadir, it_num, speclist, y, y, delta, y_bar, \
              y_bar, delta_bar, file_fancy, doprint)

