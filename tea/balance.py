#! /usr/bin/env python

############################# BEGIN FRONTMATTER ################################
#                                                                              #
#   TEA - calculates Thermochemical Equilibrium Abundances of chemical species #
#                                                                              #
#   TEA is part of the PhD dissertation work of Dr. Jasmina                    #
#   Blecic, who developed it with coding assistance from                       #
#   undergraduate M. Oliver Bowman and under the advice of                     #
#   Prof. Joseph Harrington at the University of Central Florida,              #
#   Orlando, Florida, USA.                                                     #
#                                                                              #
#   Copyright (C) 2014-2016 University of Central Florida                      #
#                                                                              #
#   This program is reproducible-research software: you can                    #
#   redistribute it and/or modify it under the terms of the                    #
#   Reproducible Research Software License as published by                     #
#   Prof. Joseph Harrington at the University of Central Florida,              #
#   either version 0.3 of the License, or (at your option) any later           #
#   version.                                                                   #
#                                                                              #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#   Reproducible Research Software License for more details.                   #
#                                                                              #
#   You should have received a copy of the Reproducible Research               #
#   Software License along with this program.  If not, see                     #
#   <http://planets.ucf.edu/resources/reproducible/>.  The license's           #
#   preamble explains the situation, concepts, and reasons surrounding         #
#   reproducible research, and answers some common questions.                  #
#                                                                              #
#   This project was started with the support of the NASA Earth and            #
#   Space Science Fellowship Program, grant NNX12AL83H, held by                #
#   Jasmina Blecic, Principal Investigator Joseph Harrington, and the          #
#   NASA Science Mission Directorate Planetary Atmospheres Program,            #
#   grant NNX12AI69G.                                                          #
#                                                                              #
#   See the file ACKNOWLEDGING in the top-level TEA directory for              #
#   instructions on how to acknowledge TEA in publications.                    #
#                                                                              #
#   We welcome your feedback, but do not guarantee support.                    #
#   Many questions are answered in the TEA forums:                             #
#                                                                              #
#   https://physics.ucf.edu/mailman/listinfo/tea-user                          #
#   https://physics.ucf.edu/mailman/listinfo/tea-devel                         #
#                                                                              #
#   Visit our Github site:                                                     #
#                                                                              #
#   https://github.com/dzesmin/TEA/                                            #
#                                                                              #
#   Reach us directly at:                                                      #
#                                                                              #
#   Jasmina Blecic <jasmina@physics.ucf.edu>                                   #
#   Joseph Harrington <jh@physics.ucf.edu>                                     #
#                                                                              #
############################## END FRONTMATTER #################################

import readconf as rc

import os
import numpy as np
from sys import argv
from sympy.core    import Symbol
from sympy.solvers import solve

import format as form

# =============================================================================
# This code produces an initial guess for the first TEA iteration by
# fulfilling the mass balance condition, sum_i(ai_j * y_i) = bj (equation (17)
# in the TEA theory paper), where i is species index, j is element index, a's
# are stoichiometric coefficients, and b's are elemental fractions by number,
# i.e., ratio of number densities of element 'j' to the total number densities
# of all elements in the system (see the end of the Section 2 in the TEA theory
# paper). The code writes the result into machine- and human-readable files.
#
# The code begins by making a directory for the output results. Then, it reads
# the header file and imports all relevant chemical data from it.
# To satisfy the mass balance equation, some yi variables remain as free
# parameters. The number of free parameters is set to the number of total
# elements in the system, thus ensuring that the mass balance equation can
# be solved for any number of input elements and output species the user
# chooses. The code locates a chunk of species (y_i) containing a sum of
# ai_j values that forbids ignoring any element in the system (sum of the
# ai_j values in a column must not be zero). This chunk is used as a set
# of free variables in the system. The initial scale for other y_i variables
# are set to a known, arbitrary number. Initially, starting values for the
# known species are set to 0.1 moles, and the mass balance equation is
# calculated. If this value does not produce all positive mole numbers,
# the code automatically sets known parameters to 10 times smaller and
# tries again. Actual mole numbers for the initial guesses of y_i are
# arbitrary, as TEA only requires a balanced starting point to initialize
# minimization. The goal of this code is to find a positive set of non-zero
# mole numbers to satisfy this requirement. Finally, the code calculates y_bar,
# initializes the iteration number, delta, and delta_bar to zero and writes
# results into machine- and human-readable output files.
#
# This code is called by runatm.py and runsingle.py and can be executed alone
# with in-shell input: balance.py <HEADER_FILE> <DIRECTORY_NAME>
# =============================================================================

# Read configuration-file parameters:
TEApars, PREATpars = rc.read()
maxiter, save_headers, save_outputs, doprint, times, \
         abun_file, location_out, xtol = TEApars

# Correct location_out name
if location_out[-1] != '/':
    location_out += '/'

# Read run-time arguments
header = argv[1:][0]              # Name of header file
desc   = argv[1:][1]              # Directory name

# Create and name outputs and results directories if they do not exist
datadir   = location_out + desc + '/outputs/' + 'transient/'

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
            free_id.append(n + m)
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

    # Make 'j' equations satisfying mass balance equation (17) in TEA theory doc:
    # sum_i(ai_j * y_i) = b_j
    eq = [[]]
    for m in np.arange(j):
        rhs = 0
        for n in np.arange(i):
            rhs += a[n, m] * y_init[n]
        rhs -= b[m]
        eq = np.append(eq, rhs)

    # Solve system of linear equations to get free y_i variables
    result = solve(list(eq), list(free), rational=False)

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

