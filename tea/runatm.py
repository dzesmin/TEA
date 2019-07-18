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

from readconf import *

# Setup for time/speed testing
if times:
    import time
    start = time.time()

import numpy as np
import sys
import ntpath
import os
import shutil
import subprocess

import format as form
import makeheader as mh
import readatm as ra

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

# Time / speed testing
if times:
    end = time.time()
    elapsed = end - start
    print("runatm.py imports:  " + str(elapsed))

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

# Correct directory names
if location_TEA[-1] != '/':
    location_TEA += '/'

if location_out[-1] != '/':
    location_out += '/'

# Retrieve pre-atm file
infile  = sys.argv[1:][0]

# If input file does not exist break
try:
    f = open(infile)
except:
    raise IOError ("\n\nPre-atmospheric file does not exist.\n")

# Retrieve current output directory name given by user
desc    = sys.argv[1:][1]

# Set up locations of necessary scripts and directories of files
inputs_dir       = location_out + desc + "/inputs/"
thermo_dir       = location_TEA + "lib/gdata"
loc_balance      = location_TEA + "tea/balance.py"
loc_iterate      = location_TEA + "tea/iterate.py"
loc_headerfile   = location_out + desc + "/headers/" + "header_" + desc + ".txt"
loc_outputs      = location_out + desc + "/outputs/"
loc_transient    = location_out + desc + "/outputs/" + "transient/"
out_dir          = location_out + desc + "/results/"
single_res       = ["results-machine-read.txt", "results-visual.txt"]

# Check if output directory already exists and inform user
if os.path.exists(inputs_dir):
    print("  Output directory " + str(inputs_dir) + " already exists.\n"
              "  Press enter to continue and overwrite existing files,\n"
              "  or quit and choose another output name.\n")

# Create directories
if not os.path.exists(inputs_dir): os.makedirs(inputs_dir)
if not os.path.exists(out_dir): os.makedirs(out_dir)
if not os.path.exists(loc_transient): os.makedirs(loc_transient)

# Check if config file exists in the working directory
TEA_config = 'TEA.cfg'
try:
    f = open(TEA_config)
except IOError:
    print("\nConfig file is missing. Place TEA.cfg in the working directory.\n")

# Inform user if TEA.cfg file already exists in inputs/ directory
if os.path.isfile(inputs_dir + TEA_config):
    print("  " + str(TEA_config) + " overwritten in inputs/ directory.")
# Copy TEA.cfg file to current inputs directory
shutil.copy2(TEA_config, inputs_dir + TEA_config)

# Inform user if abundances file already exists in inputs/ directory
head, abun_filename = ntpath.split(abun_file)
if os.path.isfile(inputs_dir + abun_filename):
    print("  " + str(abun_filename) + " overwritten in inputs/ directory.")
# Copy abundances file to inputs/ directory
shutil.copy2(abun_file, inputs_dir + abun_filename)

# Inform user if pre-atm file already exists in inputs/ directory
head, preatm_filename = ntpath.split(infile)
if os.path.isfile(inputs_dir + preatm_filename):
    print("  " + str(preatm_filename) + " overwritten in inputs/ directory.")
else:
    # Copy pre-atm file to inputs/ directory
    shutil.copy2(infile, inputs_dir + preatm_filename)

# Read pre-atm file
n_runs, spec_list, pres_arr, temp_arr, atom_arr, end_head = \
                                                 ra.readatm(infile)

# Correct species list for only species found in thermo_dir
gdata_files = os.listdir(thermo_dir)
good_spec   = []
for i in np.arange(np.size(spec_list)):
    spec_file = spec_list[i] + '.txt'
    if spec_file in gdata_files:
        good_spec = np.append(good_spec, spec_list[i])
    else:
        print('Species ' + spec_list[i] + ' does not exist in /' \
                  + thermo_dir.split("/")[-1] + ' ! IGNORED THIS SPECIES.')

# Update list of valid species
spec_list = np.copy(good_spec)

# =================== Start writing final atm file ===================
# Open final atm file for writing, keep open to add new lines 
fout_name = out_dir + desc + '.tea'
fout = open(fout_name, 'w+')

# Write a header file
header      = "# This is a final TEA output file with calculated abundances (mixing fractions) for all listed species.\n\
# Units: pressure (bar), temperature (K), abundance (unitless)."
fout.write(header + '\n\n')
fout.write('#SPECIES\n')

# Write corrected species list into pre-atm file and continue
for i in np.arange(np.size(spec_list)):
    fout.write(spec_list[i] + ' ')
fout.write("\n\n")
fout.write("#TEADATA\n")

# Write data header from the pre-atm file into each column of atm file 
fout.write(pres_arr[0].ljust(10) + ' ')
fout.write(temp_arr[0].ljust(7) + ' ')
for i in np.arange(np.size(spec_list)):
    fout.write(spec_list[i].ljust(10)+' ')
fout.write('\n')

# Times / speed check for pre-loop runtime
if times:
    new = time.time()
    elapsed = new - end
    print("pre-loop:           " + str(elapsed))

# Detect operating system for sub-process support
if os.name == 'nt': inshell = True    # Windows
else:               inshell = False   # OSx / Linux

# Allocate abundances matrix for all species and all T-Ps
abun_matrix = np.zeros(np.size(spec_list))


# ============== Execute TEA for each T-P ==============
# Loop over all lines in pre-atm file and execute TEA loop
for q in np.arange(n_runs)[1:]:

    # Print for debugging purposes
    if doprint:
        print("\nReading atm file, TP line " + str(q))
    else:
        print('\n'+ str(q))
    
    # Radius, pressure, and temp for the current line    
    pres = pres_arr[q]
    temp = temp_arr[q]
    
    # Produce header for the current line
    mh.make_atmheader(q, spec_list, \
                              pres, temp, atom_arr, desc, thermo_dir)
    
    # Time / speed testing for balance.py
    if times:
        ini = time.time()
    
    # Get balanced initial guess for the current line, run balance.py
    subprocess.call([loc_balance, loc_headerfile, desc], shell = inshell)
    
    # Retrieve balance runtime
    if times:
        fin = time.time()
        elapsed = fin - ini
        print("balance.py:         " + str(elapsed))
    
    # Execute main TEA loop for the current line, run iterate.py
    subprocess.call([loc_iterate, loc_headerfile, desc], shell = inshell)

    # Read output of TEA loop
    header, it_num, speclist, y, x, delta, y_bar, x_bar, delta_bar \
        = form.readoutput(location_out + desc + '/results/' + single_res[0])

    # Insert results from the current line (T-P) to atm file
    fout.write(pres.rjust(10) + ' ')
    fout.write(temp.rjust(7) + ' ')
    for i in np.arange(np.size(spec_list)):
        cur_abn = x[i] / x_bar
        fout.write('%1.4e'%cur_abn + ' ')
    fout.write('\n')    
   
    # Save or delete intermediate headers
    if save_headers:
        # Save header files for each T-P
        old_name = loc_headerfile
        new_name = loc_headerfile[0:-4] + "_" + '%.0f'%float(temp) + "K_" + \
                   '%1.2e'%float(pres) + "bar" + loc_headerfile[-4::]
        if os.path.isfile(new_name):
            os.remove(new_name)
        os.rename(old_name, new_name)
    else:
        # Delete header files for each T-P
        shutil.rmtree(location_out + desc + "/headers/" )
    
    # Save or delete lagrange.py, lambdacorr.py outputs and single T-P results
    if save_outputs:
        # Save directory for each T-P and its output files
        if not os.path.exists(loc_outputs): os.makedirs(loc_outputs)
        old_name = loc_transient
        new_name = loc_outputs + desc + "_" + '%.0f'%float(temp) + "K_"  \
                   '%1.2e'%float(pres) + "bar" + loc_outputs[-1::]
        if os.path.exists(new_name):
            for file in os.listdir(new_name):
                os.remove(new_name + file)
            shutil.rmtree(new_name)
        os.rename(old_name, new_name)

        # Save directory for each single T-P result and its results
        single_dir = out_dir + "results_" + '%.0f'%float(temp) + "K_" + \
                     '%1.2e'%float(pres) + "bar" + "/"
        if not os.path.exists(single_dir): os.makedirs(single_dir)
        for file in single_res:
            if os.path.isfile(single_dir + file):
                os.remove(single_dir + file)
            os.rename(out_dir + file, single_dir + file) 
    else:
        # Delete output directories and files
        shutil.rmtree(loc_outputs)
        # Delete single T-P result directories and files
        for file in single_res:
            os.remove(out_dir + file)

# Close atm file
fout.close()

# Print on-screen
print("\n  Species abundances calculated.\n  Created TEA atmospheric file.")

# Time / speed testing
if times:
    end = time.time()
    elapsed = end - start
    print("Overall run time:   " + str(elapsed) + " seconds")
