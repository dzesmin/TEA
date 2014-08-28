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

import numpy as np
from sympy.core    import Symbol
from sympy.solvers import solve

import format as form

def lagrange(it_num, datadir, doprint, direct):
    '''
    This code applies Lagrange's method and calculates minimum based on the 
    methodology elaborated in TEA Document in Section (1.2.1). Equations in
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
    direct:  object
             Object containing all of the results/data from the previous
             calculation in lagrange.py or lambdacorr.py. It is a list
             containing current header directory, current iteration 
             number, array of species names, array of initial guess, 
             array of non-corrected Lagrange values, and array of 
             lambdacorr corrected values.

    Returns
    -------
    header: string
            Name of the header file used.
    it_num: integer 
            Iteration number.
    speclist: string array
            Array containing names of molecular species. 
    y: float array
            Array containing initial guess of molecular species for
            current iteration.
    x: float array
            Array containing final mole numbers of molecular species for
            current iteration.
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

    # Read values from last iteration   
    input  = direct 

    # Take the current header file  
    header = input[0]

    # Read values from the header file
    pressure, temp, i, j, speclist, a, b, g_RT = form.readheader(header)
    
    # Use final values from last iteration (x values) as new initial
    y     = input[4] 
    y_bar = input[7] 

    # Perform checks to be safe
    it_num_check   = input[1]
    speclist_check = input[2]
    
    # Make array of checks
    check = np.array([it_num_check != it_num - 1,
            False in (speclist_check == speclist) ])
    # If iteration number given by iterate.py is not one larger as in 'direct',
    #      give error
    if check[0]:
        print("\n\nMAJOR ERROR! Read in file's it_num is not the most \
                recent iteration!\n\n")
    # If species names in the header are not the same as in 'direct', 
    #      give error 
    if check[1]:
        print("\n\nMAJOR ERROR! Read in file uses different species   \
                order/list!\n\n")      
    
        
    # ============== CREATE VARIABLES FOR LAGRANGE EQUATION ============== #

    # Create 'c' value, equation (17) TEA Document
    # ci = (g/RT)_i + ln(P)    
    c = g_RT + np.log(pressure)
    
    # Allocates array of fi(Y) over different values of i (species)
    fi_y = np.zeros(i)
    
    # Fill in fi(Y) values equation (18) TEA Document
    # fi = x_i * [ci + ln(x_i/x_bar)]
    for n in np.arange(i):
        y_frac  = np.float(y[n] / y_bar)
        fi_y[n] = y[n] * ( c[n] + np.log(y_frac) )
    
    # Allocate values of rjk. Both j and k goes from 1 to m.
    k = j 
    rjk = np.zeros((j,k))

    # Fill out values of rjk, equation (25) TEA Document 
    # rjk = rkj = sum_i(a_ij * a_ik) * y_i
    for l in np.arange(k):
        for m in np.arange(j):
            r_sum = 0.0
            for n in np.arange(i): 
                r_sum += a[n, m] * a[n, l] * y[n]      
            rjk[m, l] = r_sum
    
    # Allocate value of u, equation (27) TEA Document
    u = Symbol('u')
    
    # Allocate pi_j variables, where j is element index
    # Example: pi_2 is Lagrange multiplier of N (j = 2)
    pi = []
    for m in np.arange(j):
        name = 'pi_' + np.str(m+1)
        pi = np.append(pi, Symbol(name))
    
    # Allocate rjk * pi_j summations, equation (26) TEA Document
    # There will be j * k terms of rjk * pi_j
    sq_pi = [pi]
    for m in np.arange(j-1):
        # Make square array of pi values with shape j * k
        sq_pi = np.append(sq_pi, [pi], axis = 0) 

    # Multiply rjk * sq_pi to get array of rjk * pi_j 
    # equation (26) TEA DOcument
    rpi = rjk * sq_pi 
    

    # ======================= SET FINAL EQUATIONS ======================= #
    # Total number of equations is j + 1
    
    # Set up a_ij * fi(Y) summations equation (26) TEA Document
    # sum_i[a_ij * fi(Y)]
    aij_fiy = np.zeros((j))
    for m in np.arange(j):
        rhs = 0.0
        for n in np.arange(i):
            rhs += a[n,m] * fi_y[n]
        aij_fiy[m] = rhs
    
    # Create first j'th equations equation (26) TEA Document
    # r_1m*pi_1 + r_2m*pi_2 + ... + r_mm*pi_m + b_m*u = sum_i[a_im * fi(Y)]
    for m in np.arange(j):
        if m == 0:
            equations   = np.array([np.sum(rpi[m]) + b[m]*u - aij_fiy[m]])
        else:
            lagrange_eq = np.array([np.sum(rpi[m]) + b[m]*u - aij_fiy[m]])
            equations   = np.append(equations, lagrange_eq)

    # Last (j+1)th equation (26) TEA Document
    # b_1*pi_1 + b_2*pi_2 + ... + b_m*pi_m + 0*u = sum_i[fi(Y)]
    bpi = b * pi
    lagrange_eq_last = np.array([np.sum(bpi) - np.sum(fi_y)])
    equations = np.append(equations, lagrange_eq_last)
            
    # List all unknowns in the above set of equations
    unknowns = list(pi)
    unknowns.append(u)

    # Solve final system of j+1 equations
    fsol = solve(list(equations), unknowns)
    
    
    # ============ CALCULATE xi VALUES FOR CURRENT ITERATION ============ #
    
    # Make array of pi values
    pi_f = []
    for m in np.arange(j):
        pi_f = np.append(pi_f, [fsol[pi[m]]])

    # Calculate x_bar from solution to get 'u', equation (27) TEA Document
    # u = -1. + (x_bar/y_bar)
    u_f = fsol[u]
    x_bar = (u_f + 1.) * y_bar
    
    # Initiate array for x values of size i
    x = np.zeros(i)
    
    # Apply Lagrange solution for final set of x_i values for this iteration
    # equation (22) TEA Document
    # x_i = -fi(Y) + (y_i/y_bar) * x_bar + [sum_j(pi_j * a_ij)] * y_i
    for n in np.arange(i):
        sum_pi_aij = 0.0
        for m in np.arange(j):
            sum_pi_aij += pi_f[m] * a[n, m]
        x[n] = - fi_y[n] + (y[n]/y_bar) * x_bar + sum_pi_aij * y[n]
    
    # Calculate other variables of interest
    x_bar = np.sum(x)             # sum of all x_i
    delta = x - y                 # difference between initial and final values
    delta_bar = x_bar - y_bar     # difference between sum of initial and
                                  # final values
    
    # Name output files with corresponding iteration number name
    file       = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                         'machine-read-nocorr.txt'
    file_fancy = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                              '-visual-nocorr.txt'

    # Export all values into machine and human readable output files
    form.output(datadir, header, it_num, speclist, y, x, \
                       delta, y_bar, x_bar, delta_bar, file, doprint)
    form.fancyout(datadir, it_num, speclist, y, x, delta,\
                         y_bar, x_bar, delta_bar, file_fancy, doprint)
       
    return [header, it_num, speclist, y, x, delta, y_bar, x_bar, delta_bar]
        
