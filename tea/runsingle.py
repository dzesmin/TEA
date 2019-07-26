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

import ntpath
import os
import shutil
import subprocess
import numpy as np
import sys
import time
import six

import makeheader as mh
import readconf   as rc
import iterate    as it
import balance    as bal


location_TEA = os.path.realpath(os.path.dirname(__file__) + "/..") + "/"

# =============================================================================
# This program runs TEA over an input file that contains only one T-P.
# The code retrieves the input file and the current directory name given by the
# user. 
#
# This module prints on screen the code progress: the current iteration number, 
# and informs the user that minimization is done.
# Example:
#  100
# Maximum iteration reached, ending minimization.
#
# The program is executed with in-shell inputs:
# runsingle.py <SINGLETP_INPUT_FILE_PATH> <DIRECTORY_NAME>
# Example: ../TEA/tea/runsingle.py ../TEA/doc/examples/singleTP/inputs/singleTP_Example.txt Single_Example
# =============================================================================

# Read configuration-file parameters:
TEApars, PREATpars = rc.readcfg()
maxiter, savefiles, verb, times, abun_file, location_out, xtol, ncpu = TEApars

# Print license
print("\n\
================= Thermal Equilibrium Abundances (TEA) =================\n\
A program to calculate species abundances under thermochemical equilibrium.\n\
\n\
Copyright (C) 2014-2016 University of Central Florida.\n\
\n\
This program is reproducible-research software.  See the Reproducible\n\
Research Software License that accompanies the code, or visit:\n\
http://planets.ucf.edu/resources/reproducible\n\
Questions? Feedback? Search our mailing list archives or post a comment:\n\
https://physics.ucf.edu/mailman/listinfo/tea-user\n\
\n\
Direct contact: \n\
Jasmina Blecic <jasmina@physics.ucf.edu>        \n\
========================================================================\n")

# Correct output location name
if location_out[-1] != '/':
    location_out += '/'

# Retrieve user inputs file
infile  = sys.argv[1:][0]

# Set start time
tstart = time.time()

# Check if config file exists in current working directory
TEA_config = 'TEA.cfg'
try:
    f = open(TEA_config)
except IOError:
    print("\nConfig file is missing, place TEA.cfg in the working directory.\n")

# If input file does not exist break
try:
    f = open(infile)
except:
    raise IOError ("\nSingle T-P input file does not exist.\n")

# Retrieve current output directory name given by user
desc    = sys.argv[1:][1]

# Check if output directory exists and inform user
if os.path.exists(location_out + desc):
    six.moves.input("  Output directory " + str(location_out + desc) +  
              "/\n  already exists.\n"
              "  Press enter to continue and overwrite existing files,\n"
              "  or quit and choose another output name.\n")

# If output directory does not exist, create it
if not os.path.exists(location_out + desc):
  os.makedirs(location_out + desc)

# Set up locations of necessary files
thermo_dir     = location_TEA + "lib/gdata"

inputs_dir     = location_out + desc + "/inputs/"
loc_outputs    = location_out + desc + "/outputs/"

if savefiles:
    # Create inputs directory
    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)

    # Inform user if TEA.cfg file already exists in inputs/ directory
    if verb >= 1 and os.path.isfile(inputs_dir + TEA_config):
      print("  " + str(TEA_config) + " overwritten in inputs/ directory.")
    # Copy TEA.cfg file to current inputs directory
    shutil.copy2(TEA_config, inputs_dir + TEA_config)

    # Inform user if abundances file already exists in inputs/ directory
    head, abun_filename = ntpath.split(abun_file)
    if verb >= 1 and os.path.isfile(inputs_dir + abun_filename):
        print("  " + str(abun_filename) + " overwritten in inputs/ directory.")
    # Copy abundances file to inputs/ directory
    shutil.copy2(abun_file, inputs_dir + abun_filename)

    # Inform user if single T-P input file already exists in inputs/ directory
    if verb >= 1 and os.path.isfile(inputs_dir + infile.split("/")[-1]):
        print("  " + str(infile.split("/")[-1])
            + " overwritten in inputs/ directory.\n")
    # Copy single T-P input file to inputs directory
    shutil.copy2(infile, inputs_dir + infile.split("/")[-1])

# Times / speed check for pre-loop runtime
if times:
    tnew = time.time()
    elapsed = tnew - tstart
    print("pre-loop:           " + str(elapsed))

# Execute main TEA loop
temp, pressure, speclist = mh.read_single(infile)
free_energy, heat = mh.read_gdata(speclist, thermo_dir)
g_RT = mh.calc_gRT(free_energy, heat, temp)
stoich_arr, stoich_atom, b = mh.read_stoich(speclist, getb=True)
guess = bal.balance(stoich_arr, b, verb)

# Save info for the iteration module and remove it afterwards if neccesery
save_info = location_out, desc, speclist, temp    

# Perform iterations until reaching desired precision xtol
y, x, delta, y_bar, x_bar, delta_bar = it.iterate(pressure, stoich_arr, b,
           g_RT, maxiter, verb, times, guess, xtol, save_info)

# Save or delete lagrange.py and lambdacorr.py outputs
if savefiles:
    hfolder = location_out + desc + "/headers/"
    mh.write_header(hfolder, desc, temp, pressure, speclist,
                    stoich_atom, stoich_arr, b, g_RT)
else:
    # Results directory is automatically made when TEA is executed
    datadirr = '{:s}{:s}/results/results_{:.0f}K_{:.2e}bar/'.format(
                   location_out, desc, temp, pressure)
    # Copy
    shutil.copy2(datadirr + "results-visual.txt",
                 location_out + desc + "/results-visual.txt")
    shutil.rmtree(loc_outputs)
    shutil.rmtree('{:s}{:s}/results'.format(location_out, desc))

# Print on-screen
if verb >= 1:
  print("\n  Species abundances calculated.\n  Created results file.")

# Time / speed testing
if verb >= 1:
    tend = time.time()
    elapsed = tend - tstart
    print("Overall run time:   " + str(elapsed) + " seconds")


