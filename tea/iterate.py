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


import os
import time
import numpy as np
from sys import argv
from sys import stdout

import lagrange   as lg
import lambdacorr as lc
import format     as form
from   format import printout
import readconf as rc

# =============================================================================
# This program executes the iteration loop for TEA. It repeats Lagrangian
# minimization (lagrange.py) and lambda correction (lambdacorr.py) until the
# maximum iteration is reached. The code has time stamps for checking the speed
# of execution and is verbose for debugging purposes. Both are controlled in
# TEA.cfg file.
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
# in TEA.cfg.
#
# The program is executed by runatm.py and can be executed alone with in-shell
# input: iterate.py <HEADER_FILE> <DIRECTORY_NAME>
# =============================================================================

TEApars, PREATpars = rc.read()
maxiter, save_headers, save_outputs, doprint, times, \
         location_TEA, abun_file, location_out = TEApars

# Correct location_TEA name
if location_out[-1] != '/':
    location_out += '/'

# Read run-time arguments
header = argv[1:][0]              # Name of header file
desc   = argv[1:][1]              # Directory name

# Create and name outputs and results directories if they do not exist
datadir   = location_out + desc + '/outputs/' + 'transient/'
datadirr  = location_out + desc + '/results'

if not os.path.exists(datadir): os.makedirs(datadir)
if not os.path.exists(datadirr): os.makedirs(datadirr)

# Retrieve header info
inhead   = form.readheader(header)
pressure = inhead[0]
temp     = inhead[1]
i        = inhead[2]
j        = inhead[3]
a        = inhead[5]
b        = inhead[6]
g_RT     = inhead[7]

# Locate and read initial iteration output from balance.py
infile    = datadir + '/lagrange-iteration-0-machine-read.txt'
input     = form.readoutput(infile)

# Retrieve and set initial values
speclist  = input[2]
x         = input[3]
x_bar     = float(input[6])

# Set up first iteration
it_num  = 1
repeat  = True

# Prepare data object for iterative process
#         (see description of the 'direct' object in lagrange.py)
lambdacorr_data = x, x, 0, x_bar, x_bar, 0
info = pressure, i, j, a, b, g_RT, header, speclist

# Time / speed testing
if times:
    new = time.time()
    elapsed = new - end
    print("pre-loop setup:     " + str(elapsed))

# ====================== PERFORM MAIN TEA LOOP ====================== #

while repeat:
    # Output iteration number
    if ((not doprint) & (not times)):
        stdout.write(' {:d}\r'.format(it_num))
        stdout.flush()

    # Time / speed testing for lagrange.py
    if times:
        ini = time.time()

    # Execute Lagrange minimization
    lagrange_data = lg.lagrange(it_num, datadir, doprint, lambdacorr_data, info)

    # Time / speed testing for lagrange.py
    if times:
        fin = time.time()
        elapsed = fin - ini
        print("lagrange" + str(it_num).rjust(4) + " :      " + str(elapsed))

    # Print for debugging purposes
    if doprint:
        printout('Iteration %d Lagrange complete. Starting lambda correction...', it_num)

    # Check if x_i have negative mole numbers, if so, do lambda correction
    if np.any(lagrange_data[1] < 0):
        # Print for debugging purposes
        if doprint:
            printout('Correction required. Initializing...')

        # Time / speed testing for lambdacorr.py
        if times:
            ini = time.time()

        # Execute lambda correction
        lambdacorr_data = lc.lambdacorr(it_num, datadir, doprint, lagrange_data, info)

        # Print for debugging purposes
        if times:
            fin = time.time()
            elapsed = fin - ini
            print("lambcorr" + str(it_num).rjust(4) + " :      " + str(elapsed))

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

    # If max iteration not met, continue with next iteration cycle
    if it_num < maxiter:
        it_num += 1
        if doprint:
            printout('Max interation not met. Starting next iteration...\n')

    # ============== Stop the loop, max iteration reached ============== #

    # Stop if max iteration is reached
    else:
        printout('Maximum iteration reached, ending minimization.\n')
        # Stop the loop
        repeat = False

        # Retrieve most recent iteration values
        input_new = lambdacorr_data

        # Take most recent x_i and x_bar values
        x_new     = input_new[1]
        x_bar_new = input_new[4]

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

