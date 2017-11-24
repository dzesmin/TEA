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


def iterate(header, desc, headerfile, maxiter, doprint, times, location_out,
            guess=None, xtol=3e-6):
  """
  Run iterative Lagrangian minimization and lambda correction.

  Parameters
  ----------
  header: 1D list
  desc: String
  headerfile: String
  maxiter: Integer
     Maximum number of iterations.
  doprint: Bool
     If True, print information to screen.
  times: Bool
     If True, track excecution times.
  location_out:
  guess: 1D list
     If not None, a three-element list with input guess values for
     x, x_bar, and speclist.  Otherwise, take values from balance.py's
     output.
  xtol: Float
     Error between iterations that is acceptable for convergence.
     The routine has converged when the sum of the relative improvement
     per species becomes less than xtol, i.e.:
       sum(abs((y-x)/y)) / len(x) <= xtol.

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
  # Create and name outputs and results directories if they do not exist
  datadir   = location_out + desc + '/outputs/' + 'transient/'
  datadirr  = location_out + desc + '/results'

  if not os.path.exists(datadir): os.makedirs(datadir)
  if not os.path.exists(datadirr): os.makedirs(datadirr)

  # Retrieve header info
  pressure = header[0]
  temp     = header[1]
  i        = header[2]
  j        = header[3]
  a        = header[5]
  b        = header[6]
  g_RT     = header[7]


  # Retrieve and set initial values
  if guess is not None:
      x        = guess[0]
      x_bar    = guess[1]
      speclist = guess[2]
  else:
      # Locate and read initial iteration output from balance.py
      infile    = datadir + '/lagrange-iteration-0-machine-read.txt'
      input     = form.readoutput(infile)
      speclist = input[2]
      x        = input[3]
      x_bar    = float(input[6])

  # Prepare data object for iterative process
  #         (see description of the 'input' object in lagrange.py)
  lc_data = x, x, 0, x_bar, x_bar, 0
  info = pressure, i, j, a, b, g_RT, header, speclist

  # Time / speed testing
  if times:
      new = time.time()
      elapsed = new - end
      print("pre-loop setup:     " + str(elapsed))

  # ====================== PERFORM MAIN TEA LOOP ====================== #
  it_num  = 1
  while it_num <= maxiter:
      # Output iteration number
      if not doprint  and  not times  and  it_num%10==0:
          stdout.write(' {:d}\r'.format(it_num))
          stdout.flush()

      # Time / speed testing for lagrange.py
      if times:
          ini = time.time()

      # Execute Lagrange minimization
      lc_data = lg.lagrange(it_num, datadir, doprint, lc_data, info)

      # Time / speed testing for lagrange.py
      if times:
          fin = time.time()
          elapsed = fin - ini
          print("lagrange" + str(it_num).rjust(4) + " :      " + str(elapsed))

      # Print for debugging purposes
      if doprint:
          printout('Iteration %d Lagrange complete. Starting lambda correction...', it_num)

      # Check if x_i have negative mole numbers, if so, do lambda correction
      if np.any(lc_data[1] < 0):
          # Print for debugging purposes
          if doprint:
              printout('Correction required. Initializing...')

          # Time / speed testing for lambdacorr.py
          if times:
              ini = time.time()

          # Execute lambda correction
          lc_data = lc.lambdacorr(it_num, datadir, doprint, lc_data, info)

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
          # Print for debugging purposes
          if doprint:
              printout('Iteration %d did not need lambda correction.', it_num)

      xdiff = (lc_data[1]/lc_data[4])/(lc_data[0]/lc_data[3]) - 1
      if np.sum(np.abs(xdiff))/len(xdiff) <= xtol:
          stdout.write(' {:d}\r'.format(it_num))
          printout("The solution has converged to the given tolerance error.\n")
          break

      it_num += 1
      # If max iteration not met, continue with next iteration cycle
      if doprint:
          printout('Max interation not met. Starting next iteration...\n')


  # ============== Stop the loop, max iteration reached ============== #
  # Stop if max iteration is reached
  if it_num == maxiter+1:
      printout('Maximum iteration reached.\n')

  # Retrieve most recent iteration values
  input_new = lc_data

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
  form.output(datadirr, headerfile, it_num, speclist, x, x_new, delta,    \
              x_bar, x_bar_new, delta_bar, file_results, doprint)
  form.fancyout_results(datadirr, headerfile, it_num, speclist, x, x_new, \
                        delta, x_bar, x_bar_new, delta_bar, pressure, \
                        temp, file_fancyResults, doprint)

  return input_new
