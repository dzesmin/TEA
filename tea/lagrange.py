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

import numpy as np
from sympy.core    import Symbol
from sympy.solvers import solve

import format as form

def lagrange(it_num, datadir, doprint, input, info):
    '''
    This code applies Lagrange's method and calculates minimum based on the 
    methodology elaborated in the TEA theory document in Section (3). Equations in
    this code contain both references and an explicitly written definitions.
    The program reads the last iteration's output and data from the last header
    file, creates variables for the Lagrange equations, sets up the Lagrange
    equations, and calculates final x_i mole numbers for the current iteration 
    cycle. Note that the mole numbers that result from this function are 
    allowed to be negative. If negatives are returned, lambda correction 
    (lambdacorr.py) is necessary. The final x_i values, as well as x_bar, 
    y_bar, delta, and delta_bar are written into machine- and human-readable
    output files. This function is executed by iterate.py.

    Parameters
    ----------
    it_num:  integer 
             Iteration number.
    datadir: string
             Current directory where TEA is run.
    doprint: string
             Parameter in configuration file that allows printing for 
             debugging purposes.
    input:   List
             The input values/data from the previous
             calculation in lagrange.py or lambdacorr.py. It is a list
             containing current header directory, current iteration 
             number, array of species names, array of initial guess, 
             array of non-corrected Lagrange values, and array of 
             lambdacorr corrected values.
    info:    List
             FINDME

    Returns
    -------
    y: float array
            Input guess of molecular species.
    x: float array
            Final mole numbers of molecular species.
    delta: float array
            Array containing change in initial and final mole numbers of
            molecular species for current iteration.
    y_bar: float
            Array containing total sum of initial guesses of all molecular 
            species for current iteration.
    x_bar: float
            Total sum of the final mole numbers of all molecular species.
    delta_bar: float 
            Change in total of initial and final mole numbers of molecular
            species.
    '''
    # Read values from the header file
    pressure = info[0]
    i        = info[1]
    j        = info[2]
    a        = info[3]
    b        = info[4]
    g_RT     = info[5]
    header   = info[6]
    speclist = info[7]

    # Use final values from last iteration (x values) as new initial
    y     = input[1]
    y_bar = input[4]

    # ============== CREATE VARIABLES FOR LAGRANGE EQUATION ============== #

    # Create 'c' value, equation (18) TEA theory document
    # ci = (g/RT)_i + ln(P)    
    c = g_RT + np.log(pressure)
    
    # Fill in fi(Y) values equation (19) TEA theory document
    # fi = x_i * [ci + ln(x_i/x_bar)]
    fi_y = y * (c + np.log(y / y_bar))

    # Allocate values of rjk. Both j and k goes from 1 to m.
    k = j 
    rjk = np.zeros((j,k))

    # Fill out values of rjk, equation (26) TEA theory document 
    # rjk = rkj = sum_i(a_ij * a_ik) * y_i
    for l in np.arange(k):
        for m in np.arange(j):
            rjk[m, l] = np.sum(a[:,m] * a[:,l] * y)
    
    # Allocate value of u, equation (28) TEA theory document
    u = Symbol('u')
    
    # Allocate pi_j variables, where j is element index
    # Example: pi_2 is Lagrange multiplier of N (j = 2)
    pi = np.empty(j, dtype=object)
    for m in np.arange(j):
        pi[m] = Symbol('pi_' + str(m+1))

    # Allocate rjk * pi_j summations, equation (27) TEA theory document
    # There will be j * k terms of rjk * pi_j
    # Make square array of pi values with shape j * k
    sq_pi = np.tile(pi, (j,1))

    # Multiply rjk * sq_pi to get array of rjk * pi_j 
    # equation (27) TEA theory document
    rpi = rjk * sq_pi 

    # ======================= SET FINAL EQUATIONS ======================= #
    # Total number of equations is j + 1
    
    # Set up a_ij * fi(Y) summations equation (27) TEA theory document
    # sum_i[a_ij * fi(Y)]
    aij_fiy = np.sum(a.T*fi_y, axis=1)
    
    # Create first j'th equations equation (27) TEA theory document
    # r_1m*pi_1 + r_2m*pi_2 + ... + r_mm*pi_m + b_m*u = sum_i[a_im * fi(Y)]
    for m in np.arange(j):
        if m == 0:
            equations   = np.array([np.sum(rpi[m]) + b[m]*u - aij_fiy[m]])
        else:
            lagrange_eq = np.array([np.sum(rpi[m]) + b[m]*u - aij_fiy[m]])
            equations   = np.append(equations, lagrange_eq)

    # Last (j+1)th equation (27) TEA theory document
    # b_1*pi_1 + b_2*pi_2 + ... + b_m*pi_m + 0*u = sum_i[fi(Y)]
    bpi = b * pi
    lagrange_eq_last = np.array([np.sum(bpi) - np.sum(fi_y)])
    equations = np.append(equations, lagrange_eq_last)
            
    # List all unknowns in the above set of equations
    unknowns = list(pi)
    unknowns.append(u)

    # Solve final system of j+1 equations
    fsol = solve(list(equations), unknowns, rational=False)
    
    
    # ============ CALCULATE xi VALUES FOR CURRENT ITERATION ============ #
    # Make array of pi values
    pi_f = np.zeros(j, np.double)
    for m in np.arange(j):
        pi_f[m] = fsol[pi[m]]
    # Calculate x_bar from solution to get 'u', equation (28) TEA theory document
    # u = -1. + (x_bar/y_bar)
    x_bar = (fsol[u] + 1.0) * y_bar
    
    # Apply Lagrange solution for final set of x_i values for this iteration
    # equation (23) TEA theory document
    # x_i = -fi(Y) + (y_i/y_bar) * x_bar + [sum_j(pi_j * a_ij)] * y_i
    sum_pi_aij = np.sum(pi_f*a, axis=1)
    x = np.array(-fi_y + (y/y_bar)*x_bar + sum_pi_aij*y, np.double)

    # Calculate other variables of interest
    x_bar = np.sum(x)             # sum of all x_i
    delta = x - y                 # difference between initial and final values
    delta_bar = x_bar - y_bar     # difference between sum of initial and
                                  # final values

    # Name output files with corresponding iteration number name
    if doprint:
      file       = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                         'machine-read-nocorr.txt'
      file_fancy = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                              '-visual-nocorr.txt'
      # Export all values into machine and human readable output files
      form.output(datadir, header, it_num, speclist, y, x, \
                       delta, y_bar, x_bar, delta_bar, file, doprint)
      form.fancyout(datadir, it_num, speclist, y, x, delta,\
                         y_bar, x_bar, delta_bar, file_fancy, doprint)
       
    return y, x, delta, y_bar, x_bar, delta_bar
