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

# Setup for time/speed testing
if times:
    import time
    start = time.time()

import os
from numpy import size
from numpy import where
from sys import argv
from sys import stdout

import lagrange   as lg
import lambdacorr as lc
import format     as form
from   format import printout

# =============================================================================
# This program executes the iteration loop for TEA. It repeats Lagrangian 
# minimization (lagrange.py) and lambda correction (lambdacorr.py) until the
# maximum iteration is reached. The code has time stamps for checking the speed
# of execution and is verbose for debugging purposes. Both are controlled in
# TEA_config.py file.
#
# The flow of the code goes as follows: the current header, output, and result
# directory are read; physical properties are retrieved from the header, and 
# the balance.py output is read as the initial iteration input and passed to
# lagrange.py. Lagrange x_i values are then checked for negative values: the
# next iteration starts either with lambda correction output (if negative x_i's
# are found) or with the output produced by lagrange.py (if all x_i's are
# positive). This procedure is repeated until the maximum iteration is reached,
# which stops the loop. Intermediate results from each iteration step are
# written in the machine- and human-readable output files on the user's request
# in TEA_config.py. 
#
# The program is executed by runatm.py and can be executed alone with in-shell
# input: iterate.py <header file> <name of the result directory>
# =============================================================================

# Time / speed testing
if times:
    end = time.time()
    elapsed = end - start
    print("iterate.py imports: " + str(elapsed))

# Read run-time arguments
header = argv[1:][0]              # Name of header file
desc   = argv[1:][1]              # Description of the run

# Create and name outputs and results directories if they do not exist
datadir   = 'outputs/' + 'transient/' + desc # Location of storage directory
datadirr  = 'results/' + desc                # Location of final results

if not os.path.exists(datadir): os.makedirs(datadir)
if not os.path.exists(datadirr): os.makedirs(datadirr)

# Retrieve header info
inhead    = form.readheader(header)
pressure  = inhead[0]
temp      = inhead[1]

# Locate and read initial iteration output from balance.py
infile    = datadir + '/lagrange-iteration-0-machine-read.txt'
input     = form.readoutput(infile)

# Retrieve and set initial values
speclist  = input[2]
x         = input[3]
x_bar     = input[6]

# Set up first iteration 
it_num  = 1
repeat  = True

# Prepare data object for iterative process 
#         (see description of the 'direct' object in lagrange.py)
lambdacorr_data = [header, 0, speclist, x, x, 0, x_bar, x_bar, 0]

# Time / speed testing
if times:
    new = time.time()
    elapsed = new - end
    print("pre-loop setup:     " + str(elapsed))

# ====================== PERFORM MAIN TEA LOOP ====================== #

while repeat:
    # Output iteration number
    if ((not doprint) & (not times)):
        stdout.write(' ' + str(it_num) + '\r')
        stdout.flush()

    # Time / speed testing for lagrange.py
    if times:
        ini = time.time()
    
    # Execute Lagrange minimization
    lagrange_data = lg.lagrange(it_num, datadir, doprint, lambdacorr_data)
    
    # Time / speed testing for lagrange.py
    if times:
        fin = time.time()
        elapsed = fin - ini
        print("lagrange" + str(it_num).rjust(4) + " :      " + str(elapsed))    
      
    # Print for debugging purposes
    if doprint:
        printout('Iteration %d Lagrange complete. Starting lambda correction...', it_num)
    
    # Take final x_i mole numbers from last Lagrange calculation 
    lagrange_x = lagrange_data[4]
    
    # Check if x_i have negative mole numbers, and if yes perform lambda correction
    if where((lagrange_x < 0) == True)[0].size != 0:
        # Print for debugging purposes 
        if doprint:
            printout('Correction required. Initializing...')
            
        # Time / speed testing for lambdacorr.py
        if times:
            ini = time.time()
        
        # Execute lambda correction
        lambdacorr_data = lc.lambdacorr(it_num, datadir, doprint, \
                                                   lagrange_data)
        
        # Print for debugging purposes
        if times:
            fin = time.time()
            elapsed = fin - ini
            print("lambcorr" + str(it_num).rjust(4) + " :      " + \
                                                      str(elapsed))
        
        # Print for debugging purposes
        if doprint:
            printout('Iteration %d lambda correction complete. Checking precision...', it_num)

    # Lambda correction is not needed
    else:
        # Pass previous Lagrange results as inputs to next iteration
        lambdacorr_data = lagrange_data

        # Print for debugging purposes
        if doprint:
            printout('Iteration %d did not need lambda correction.', it_num)
    
    # Retrieve most recent iteration values
    input_new = lambdacorr_data

    # Take most recent x_i and x_bar values    
    x_new     = input_new[4]
    x_bar_new = input_new[7]
    
    # If max iteration not met, continue with next iteration cycle
    if it_num < maxiter: 
        # Add 1 to iteration number
        it_num += 1

        # Print for debugging purposes
        if doprint:
            printout('Max interation not met. Starting next iteration...\n')
    
    # ============== Stop the loop, max iteration reached ============== #

    # Stop if max iteration is reached 
    else:
        # Print to screen
        printout('Maximum iteration reached, ending minimization.\n')

        # Set repeat to False to stop the loop
        repeat = False

        # Calculate delta values
        delta = x_new - x

        # Calculate delta_bar values
        delta_bar = x_bar_new - x_bar
        
        # Name output files with corresponding iteration number name
        file_results       = datadirr + '/results-machine-read.txt'
        file_fancyResults  = datadirr + '/results-visual.txt'

        # Export all values into machine and human readable output files  
        form.output(datadirr, header, it_num, speclist, x, x_new, delta,    \
                    x_bar, x_bar_new, delta_bar, file_results, doprint)
        form.fancyout_results(datadirr, header, it_num, speclist, x, x_new, \
                              delta, x_bar, x_bar_new, delta_bar, pressure, \
                              temp, file_fancyResults, doprint)

