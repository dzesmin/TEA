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
import sys
import ntpath
import os
import shutil
import time
import multiprocessing as mp
import ctypes
import warnings

import readconf as rc
import iterate  as it
import format   as form
import makeheader as mh
import readatm  as ra
import balance  as bal

location_TEA = os.path.realpath(os.path.dirname(__file__) + "/..") + "/"


# =============================================================================
# This program runs TEA over a pre-atm file that contains multiple T-P points.
# The code first retrieves the pre-atm file, and the current directory
# name given by the user. Then, it sets locations of all necessary modules
# and directories of files that will be used. It allocates an array to store
# the final abundances for each species of each T-P run. The program loops over
# all lines (T-P points) in the pre-atm file and executes the modules in the
# following order: readatm.py, makeheader.py, balance,py, iterate.py, and
# readoutput.py. Iterate.py executes lagrange.py and lambdacorr.py.
# Readoutput.py reads results from the TEA iteration loop executed in
# iterate.py. The code then opens the final atm file to write the results.
# It takes first common lines from the pre-atm file and writes the data from
# the stored pressure, temperature, and abundances array. The code has a
# condition to save or delete all intermediate files, time stamps for checking
# the speed of execution, and is verbose for debugging purposes. If these files
# are saved, the function will create a unique directory for each T-P point.
# This functionality is controlled in the TEA.cfg file. The final results with
# the input and the configuration files are saved in the results/ directory.
# The pre-atmospheric file, config file, and abundances file used for this run
# are copied to inputs/ directory.
#
# This module prints on screen the current T-P line from the pre-atm file, the
# current iteration number, and informs the user that minimization is done.
# Example:
# 5
#  100
# Maximum iteration reached, ending minimization.
#
# The program is executed with in-shell inputs:
# runatm.py <MULTITP_INPUT_FILE_PATH> <DIRECTORY_NAME>
# Example: ../TEA/tea/runatm.py ../TEA/tea/doc/examples/multiTP/atm_inputs/multiTP_Example.atm example_multiTP
# =============================================================================

def worker(pressure, temp, b, free_energy, heat, stoich_arr, guess,
           maxiter, verb, times, xtol, savefiles, start, end, abn):
    """
    Multiprocessing thermochemical-equilibrium calculation.
    """
    # Switch off verbosity if using more than one CPU:
    if ncpu > 1:
        verb, times = 0, False

    save_info = None
    for q in np.arange(start, end):
        if verb >= 1:
            print('\nLayer {:d}:'.format(q))
        g_RT = mh.calc_gRT(free_energy, heat, temp[q])
        if savefiles:
            save_info = location_out, desc, speclist, temp[q]
            hfolder = location_out + desc + "/headers/"
            mh.write_header(hfolder, desc, temp[q], pressure[q], speclist,
                            atom_name, stoich_arr, b[q], g_RT)

        # Execute main TEA loop for the current line, run iterate.py
        y, x, delta, y_bar, x_bar, delta_bar = it.iterate(pressure[q],
          stoich_arr, b[q], g_RT, maxiter, verb, times, guess, xtol, save_info)
        guess = x, x_bar
        abn[q] = x/x_bar


# Read configuration-file parameters:
TEApars, PREATpars = rc.read()
maxiter, savefiles, verb, times, abun_file, location_out, xtol, ncpu = TEApars

# Print license
if verb >= 1:
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

# Correct directory names
if location_out[-1] != '/':
    location_out += '/'

# Retrieve pre-atm file
infile  = sys.argv[1:][0]
# Retrieve current output directory name given by user
desc    = sys.argv[1:][1]

# Check if config file exists in the working directory
TEA_config = 'TEA.cfg'
try:
    f = open(TEA_config)
except IOError:
    print("\nConfig file is missing. Place TEA.cfg in the working directory.\n")

# If input file does not exist break
try:
    f = open(infile)
except:
    raise IOError ("\n\nPre-atmospheric file does not exist.\n")

# Set up locations of necessary scripts and directories of files
thermo_dir = location_TEA + "lib/gdata"


if savefiles:
  inputs_dir = location_out + desc + "/inputs/"
  out_dir    = location_out + desc + "/results/"

  # Check if output directory already exists and inform user:
  if os.path.exists(inputs_dir):
      raw_input("  Output directory " + str(inputs_dir) + " already exists.\n"
                "  Press enter to continue and overwrite existing files,\n"
                "  or quit and choose another output name.\n")
  # Create directories
  if not os.path.exists(inputs_dir):
      os.makedirs(inputs_dir)
  # Copy TEA.cfg file to current inputs directory:
  shutil.copy2(TEA_config, inputs_dir + TEA_config)
  # Copy abundances file to inputs/ directory:
  head, abun_filename = ntpath.split(abun_file)
  shutil.copy2(abun_file,  inputs_dir + abun_filename)
  # Copy pre-atm file to inputs/ directory:
  head, preatm_filename = ntpath.split(infile)
  shutil.copy2(infile,     inputs_dir + preatm_filename)


# Read pre-atm file
n_runs, speclist, pres_arr, temp_arr, atom_arr, atom_name, end_head = \
                                                 ra.readatm(infile)

# Number of output species:
nspec = np.size(speclist)

# Correct species list for only species found in thermo_dir
gdata_files = os.listdir(thermo_dir)
good_spec   = []
for i in np.arange(nspec):
    spec_file = speclist[i] + '.txt'
    if spec_file in gdata_files:
        good_spec = np.append(good_spec, speclist[i])
    else:
        print('Species ' + speclist[i] + ' does not exist in /' \
                  + thermo_dir.split("/")[-1] + ' ! IGNORED THIS SPECIES.')

# Update list of valid species
speclist = np.copy(good_spec)

# =================== Start writing final atm file ===================
# Open final atm file for writing, keep open to add new lines
fout_name = desc + '.tea'
if savefiles:
    fout_name = out_dir + desc + '.tea'

# FINDME: If fout is None, return array
fout = open(fout_name, 'w+')

# Write a header file
fout.write(
    "# This is a final TEA output file with calculated abundances (mixing "
    "fractions) for all listed species."
    "\n# Units: pressure (bar), temperature (K), abundance (unitless).\n\n")
fout.write('#SPECIES\n')

# Write corrected species list into pre-atm file and continue
for i in np.arange(nspec):
    fout.write(speclist[i] + ' ')
fout.write("\n\n")
fout.write("#TEADATA\n")

# Write data header from the pre-atm file into each column of atm file
fout.write('#Pressure'.ljust(10) + ' ')
fout.write('Temp'.ljust(7) + ' ')
for i in np.arange(nspec):
    fout.write(speclist[i].ljust(10)+' ')
fout.write('\n')

# Times / speed check for pre-loop runtime
if times:
    new = time.time()
    elapsed = new - end
    print("pre-loop:           " + str(elapsed))

# supress warning that ctypeslib will throw:
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  # Allocate abundances matrix for all species and all T-Ps
  sm_abn = mp.Array(ctypes.c_double, n_runs*nspec)
  abn = np.ctypeslib.as_array(sm_abn.get_obj()).reshape((n_runs, nspec))

# Bound ncpu to the manchine capacity:
ncpu = np.clip(ncpu, 1, mp.cpu_count())
chunksize = int(n_runs/float(ncpu)+1)

# Load gdata:
free_energy, heat = mh.read_gdata(speclist, thermo_dir)
stoich_arr, elem_arr = mh.read_stoich(speclist)

temp_arr = np.array(temp_arr, np.double)
pres_arr = np.array(pres_arr, np.double)
atom_arr = np.array(atom_arr, np.double)

# Use only elements with non-null stoichiometric values:
eidx = np.in1d(atom_name, elem_arr)
atom_arr = atom_arr[:,eidx]

# Time / speed testing for balance.py
if times:
    ini = time.time()
# Initial abundances guess:
guess = bal.balance(stoich_arr, atom_arr[0], verb)
# Retrieve balance runtime
if times:
    fin = time.time()
    elapsed = fin - ini
    print("balance.py:         " + str(elapsed))

# ============== Execute TEA for each T-P ==============
# Loop over all lines in pre-atm file and execute TEA loop
processes = []
for n in np.arange(ncpu):
  start = n * chunksize
  end   = np.amin(((n+1) * chunksize, n_runs))
  proc = mp.Process(target=worker, args=(pres_arr, temp_arr, atom_arr,
           free_energy, heat, stoich_arr, guess, maxiter, verb, times,
           xtol, savefiles, start, end, abn))
  processes.append(proc)
  proc.start()
# Make sure all processes finish their work:
for n in np.arange(ncpu):
  processes[n].join()


# Write output:
for q in np.arange(n_runs):
    # Insert results from the current line (T-P) to atm file
    fout.write("{:.5e} {:.5f} ".format(pres_arr[q], temp_arr[q]))
    for i in np.arange(nspec):
        fout.write('{:1.4e} '.format(abn[q,i]))
    fout.write('\n')

# Close atm file
fout.close()

# Print on-screen
if verb >= 1:
  print("\n  Species abundances calculated.\n  Created TEA atmospheric file.")

# Time / speed testing
if times:
    end = time.time()
    elapsed = end - start
    print("Overall run time:   " + str(elapsed) + " seconds")
