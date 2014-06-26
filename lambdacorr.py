#! /usr/bin/env python

# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
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
import numpy as np
import format as form


def lambdacorr(it_num, datadir, doprint, direct):
    '''
    This module applies lambda correction method (see Section 1.3 in TEA 
    Document). When input mole numbers are negative, the code corrects them to
    positive values and pass them to the next iteration cycle. The code reads
    the values from the last lagrange output, the information from the header
    file, performs checks, and starts setting basic equations. It sets a smart
    range so it can efficiently explore the lambda values from [0,1]. Half of 
    the range is sampled exponentially, and the other half linearly, totalling
    150 points. The code retrieves the last lambda value before first 
    derivative becomes positive (equation (33) in TEA Document), and corrects
    negative mole numbers to positive.

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
    header:  string
             Name of the header file used.
    it_num:  integer 
             Iteration number.
    speclist: array of strings
             Array containing names of molecular species. 
    y: array of floats
             Array containing initial guess of molecular species for
             current iteration.
    x_corr:  array of floats
             Array containing final mole numbers of molecular species for
             current iteration.
    delta_corr: array of floats
             Array containing change of initial and final mole numbers of 
             molecular species for current iteration.
    y_bar: float
             Array containing total initial guess of all molecular species for
             current iteration.
    x_corr_bar: float
             Total sum of the final mole numbers of all molecular species.
    detla_corr_bar: float 
             Change in total number of all species.
    doprint: string
             Parameter in configuration file that allows printing for 
             debugging purposes.

    Notes
    -----
    The code works without adjustments and with high precision for the
    the fractional abundances (mixing fractions) up to 10e-14 and the
    temperature range of 1000 - 4000 K. For temperatures below 1000 K and 
    mixing fractions below 10e-14, the code produces results with low 
    precision. To improve the precision, adjust the lambda exploration
    variables 'lower' and 'steps' to larger magnitudes (i.e., lower = -100,
    steps = 1000). This will lengthen the time of execution.
    '''

    # Suppress nan warnings, as they are used for finding valid minima
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    
     # Read values from last lagrange output
    input  = direct

    # Take the current header file
    header = input[0]

    # Read values from the header file
    pressure, temp, i, j, speclist, a, b, g_RT = form.readheader(header)
   
    # Take final values from last iteration, lagrange.py
    y         = input[3]
    x         = input[4]
    delta     = input[5]
    y_bar     = input[6]
    x_bar     = input[7]
    delta_bar = input[8]

    # Perform checks to be safe
    it_num_check   = input[1]
    speclist_check = input[2]
    
    # Make array of checks
    check = np.array([it_num_check != it_num,
                      False in (speclist_check == speclist) ])

    # If iteration number given by iterate.py is not one larger as in 'direct',
    #      give error
    if check[0]:
        print("\n\nMAJOR ERROR! Read in file's it_num is not the    \
                current iteration!\n\n")
    # If species names in the header are not the same as in 'direct', 
    #      give error 
    if check[1]:
        print("\n\nMAJOR ERROR! Read in file uses different species \
                order/list!\n\n")
    
    # Create 'c' value, equation (16) TEA Document
    # c_i = (g/RT)i + ln(P)    
    c = g_RT + np.log(pressure)
    
    # Set equation (33) TEA Document
    # dF(lam)/dlam = sum_i delta_i[(g(T)/RT)_i + lnP + 
    #                ln (yi+lam*delta_i)/(y_bar+lam*delta_bar)]
    def dF_dlam(s, i, x, y, delta, c, x_bar, y_bar, delta_bar):
        dF_dlam = 0
        for n in np.arange(i):
            dF_dlam += delta[n] * (c[n] + np.log(y[n] + s*delta[n]) - \
                       np.log(y_bar + s*delta_bar))
        return dF_dlam
    
    # Create the range of lambda values to explore. To speed up finding
    #        the correct lambda value to use, the range is split into two
    #        parts at range_split: the lower exponential range and the 
    #        higher linear range. This helps the system to converge faster.
    range_split = 0.5

    # Create exponential range, low_range
    # Exponent parameter that gives a value close to zero for the start of
    #          lambda exploration
    lower       = -50

    # Define number of steps to explore exponential range
    steps       = 100

    # Create lower exponential range
    low_range = np.exp(np.linspace(lower, 0, steps+1))

    # Create linear, evenly spaced range, high_range
    high_step = 0.01
    high_range = np.arange(0.5, 1 + high_step, high_step)

    # Combine the two ranges to create one overall range for lambda exploration
    smart_range = np.append(low_range[low_range <= range_split], high_range)

    # Set that lambda is not found
    lam_not_found = True

    # Retrieve last lambda value explored before the minimum energy is passed
    for h in smart_range:
        val = dF_dlam(h, i, x, y, delta, c, x_bar, y_bar, delta_bar)
        if val > 0 or np.isnan(val) == True:
            break

        # If lambda found, take lambda and set lambda not found to false
        lam = h
        lam_not_found = False

    # If lambda is not found (break), function F (equation (35) in TEA
    #    Document) set the final x mole numbers to the values calculated
    #    in the last iteration output 
    if lam_not_found:
        x_corr = y
    else:
        x_corr = y + lam * delta

    # Correct x values given this value of lambda
    x_corr_bar = np.sum(x_corr)
    delta_corr = y - x_corr
    delta_corr_bar = x_corr_bar - y_bar
    
    # Name output files with corresponding iteration number name
    file       = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                           '-machine-read.txt'
    file_fancy = datadir + '/lagrange-iteration-' + np.str(it_num) + \
                                                 '-visual.txt'

    # Export all values into machine and human readable output files
    form.output(datadir, header, it_num, speclist, y, x_corr, delta_corr,  \
                    y_bar, x_corr_bar, delta_corr_bar, file, doprint)
    form.fancyout(datadir, it_num, speclist, y, x_corr, delta_corr, y_bar, \
                     x_corr_bar, delta_corr_bar, file_fancy, doprint)
        
    return [header, it_num, speclist, y, x_corr, delta_corr, y_bar,       \
                                x_corr_bar, delta_corr_bar, doprint]

