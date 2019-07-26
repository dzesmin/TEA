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


# =============================================================================
# This program executes the iteration loop for TEA. It repeats Lagrangian
# minimization (lagrange.py) and lambda correction (lambdacorr.py) until the
# set precision level (tolerance error) or maxium iteration is reached. 
# The code has time stamps for checking the speed of execution and 
# is verbose for debugging purposes. Both are controlled in TEA.cfg file.
#
# The flow of the code goes as follows: the current header, output, and result
# directory are read; physical properties are retrieved from the header, and
# the balance.py output is read as the initial iteration input and passed to
# lagrange.py. Lagrange x_i values are then checked for negative values: the
# next iteration starts either with lambda correction output (if negative x_i's
# are found) or with the output produced by lagrange.py (if all x_i's are
# positive). This procedure is repeated until the set precision level
# (tolerance error) or maxium iteration is reached which stops the loop. 
# Intermediate results from each iteration step are written in the 
# machine- and human-readable output files on the user's request in TEA.cfg.
#
# The program is executed by runatm.py and runsingle.py
# =============================================================================

def iterate(pressure, a, b, g_RT, maxiter, verb, times, guess, xtol=1e-8,
            save_info=None):
    """
    Run iterative Lagrangian minimization and lambda correction.

    Parameters
    ----------
    pressure: Float
          Atmospheric pressure (bar).
    a: 2D float ndarray
          Species stoichiometric coefficients.
    b: 1D float ndarray
          Elemental mixing fractions.
    g_RT: 1D float ndarray
          Species chemical potentials.
    maxiter: Integer
          Maximum number of iterations.
    verb: Integer
          Verbosity level (0=mute, 1=quiet, 2=verbose).
    times: Bool
          If True, track excecution times.
    guess: 1D list
          A two-element list with input guess values for x and x_bar.
    xtol: Float
          Error between iterations that is acceptable for convergence.
          The routine has converged when the sum of the relative improvement
          per species becomes less than xtol, i.e.:
         sum(abs((y-x)/y)) / len(x) <= xtol.
    save_info: List of stuff
          If not None, save info to files.  The list contains:
           [location_out, desc, speclist, temp]

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
    """

    # Retrieve header info
    i, j = np.shape(a)

    # Retrieve and set initial values
    x, x_bar = guess

    # Prepare data object for iterative process
    #         (see description of the 'input' object in lagrange.py)
    lc_data = x, x, 0, x_bar, x_bar, 0
    info = pressure, i, j, a, b, g_RT

    # ====================== PERFORM MAIN TEA LOOP ====================== #
    it_num  = 1
    while it_num <= maxiter:
        # Output iteration number
        if verb >= 1  and  (it_num%10) == 0:
            stdout.write(' {:d}\r'.format(it_num))
            stdout.flush()

        # Time / speed testing for lagrange.py
        if times:
            ini = time.time()

        # Execute Lagrange minimization:
        lc_data = lg.lagrange(it_num, verb, lc_data, info, save_info)

        # Time / speed testing for lagrange.py
        if times:
            fin = time.time()
            elapsed = fin - ini
            print("lagrange {:>4d}:  {:f} s.".format(it_num, elapsed))

        # Print for debugging purposes
        if verb > 1:
            printout("Iteration {:d} Lagrange complete.  Starting lambda "
                   "correction...".format(it_num))

        # Check if x_i have negative mole numbers, if so, do lambda correction
        if np.any(lc_data[1] < 0):
            # Print for debugging purposes
            if verb > 1:
                printout('Correction required. Initializing...')

            # Time / speed testing for lambdacorr.py
            if times:
                ini = time.time()

            # Execute lambda correction
            lc_data = lc.lambdacorr(it_num, verb, lc_data, info, save_info)

            # Print for debugging purposes
            if times:
                fin = time.time()
                elapsed = fin - ini
                print("lambcorr {:>4d}:  {:f} s.".format(it_num, elapsed))

            # Print for debugging purposes
            if verb > 1:
                printout("Iteration {:d} lambda correction complete.  "
                         "Checking precision...".format(it_num))

        # Lambda correction is not needed
        else:
            # Print for debugging purposes
            if verb > 1:
                printout('Iteration {:d} did not need lambda correction.'.
                        format(it_num))

        # Check the tolerance
        xdiff = (lc_data[1]/lc_data[4])/(lc_data[0]/lc_data[3]) - 1
        if np.sum(np.abs(xdiff))/len(xdiff) <= xtol:
            if verb >= 1:
                stdout.write(' {:d}\r'.format(it_num))
                printout("The solution has converged to the given tolerance error.\n")
            break

        it_num += 1
        # If max iteration not met, continue with next iteration cycle
        if verb > 1:
            printout('Max interation not met. Starting next iteration...\n')


    # ============== Stop the loop, max iteration reached ============== #

    # Stop if max iteration is reached
    if verb >= 1 and it_num == maxiter+1:
        printout('Maximum iteration reached, ending minimization.\n')

    # Retrieve most recent iteration values
    input_new = lc_data

    # Take most recent x_i and x_bar values
    x_new     = input_new[1]
    x_bar_new = input_new[4]

    # Calculate delta values
    delta = x_new - x

    # Calculate delta_bar values
    delta_bar = x_bar_new - x_bar

    if save_info is not None:
        location_out, desc, speclist, temp = save_info
        hfolder = location_out + desc + "/headers/"
        headerfile = "{:s}/header_{:s}_{:.0f}K_{:.2e}bar.txt".format(
                      hfolder, desc, temp, pressure)

        # Create and name outputs and results directories if they do not exist
        datadirr = '{:s}{:s}/results/results_{:.0f}K_{:.2e}bar'.format(
                   location_out, desc, temp, pressure)
        if not os.path.exists(datadirr):
            os.makedirs(datadirr)

        # Export all values into machine and human readable output files
        file = "{:s}/results-machine-read.txt".format(datadirr)
        form.output(headerfile, it_num, speclist, x, x_new, delta,
                x_bar, x_bar_new, delta_bar, file, verb)
        file = "{:s}/results-visual.txt".format(datadirr)
        form.fancyout_results(headerfile, it_num, speclist, x, x_new, delta,
                x_bar, x_bar_new, delta_bar, pressure, temp, file, verb)

    return input_new


