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

import numpy as np
import sys
import os
import shutil
import subprocess

import format as form
import makeheader as mh
import readatm as ra

# =============================================================================
# This program runs TEA over a pre-atm file that contains multiple T-P's.
# The code first retrieves the pre-atm file, and the current directory
# name given by the user. Then, it sets locations of all necessary modules
# and directories of files that will be used. The program loops over all lines
# (T-P's) in the pre-atm file and writes results in the final atm file. It 
# executes the modules in the following order: readatm.py, makeheader.py, 
# balance,py, iterate.py, and readoutput.py. Iterate.py executes lagrange.py 
# and lambdacorr.py. Readoutput() reads results from the TEA iteration loop 
# executed in iterate.py and pass them for writing in the atm file. The code 
# has a condition to save or delete all intermediate files, time stamps
# for checking the speed of execution, and is verbose for debugging purposes.
# If these files are saved, the function will create a unique directory for
# each T-P point. This functionality is controlled in TEA_config file. The 
# final results with the input and the configuration files are saved in the 
# results/ directory.
#
# This module prints on screen the code progress: the current T-P line from the 
# pre-atm file, the current iteration number, and informs the user that
# minimization is done.
# Example:
# 5
#  100
# Maximum iteration reached, ending minimization.
#
# The program is executed with in-shell inputs:
# runatm.py <pre-atm file> <name of the result directory>
# Example: runatm.py inputs/Examples/atm_Example.dat Atm_Example
# =============================================================================

# Time / speed testing
if times:
    end = time.time()
    elapsed = end - start
    print("runatm.py imports:  " + str(elapsed))

# Print license
print("\n\
================= Thermal Equilibrium Abundances (TEA) =================\n\
A code to calculate gaseous molecular abundances for hot-Jupiter \n\
atmospheres under thermochemical equilibrium conditions. \n\
Copyright (C) 2014 University of Central Florida.  All rights reserved. \n\
Test version, not for redistribution.  \n\
For feedback, contact: \n\
Jasmina Blecic <jasmina@physics.ucf.edu>        \n\
Joseph Harrington <jh@physics.ucf.edu>          \n\
========================================================================\n")

# Retrieve pre-atm file
infile  = sys.argv[1:][0]

# Retrieve current output directory name given by user
desc    = sys.argv[1:][1]

# Set up locations of necessary scripts and directories of files
cwd = os.getcwd() + '/'
thermo_dir       = cwd + "gdata"
loc_readatm      = cwd + "readatm.py"
loc_makeheader   = cwd + "makeheader.py"
loc_balance      = cwd + "balance.py"
loc_iterate      = cwd + "iterate.py"
loc_headerfile   = cwd + "/headers/" + desc + "/header_" + desc + ".txt"
loc_outputs      = cwd + "/outputs/" + desc + "/"
loc_transient    = cwd + "/outputs/" + "transient/"
loc_outputs_temp = loc_transient + desc + "/"
out_dir          = cwd + "/results/" + desc + "/"
single_res       = ["results-machine-read.txt", "results-visual.txt"]

# Read pre-atm file
n_runs, spec_list, radi_arr, pres_arr, temp_arr, atom_arr, end_head = \
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

# Open final atm file for writing, keep open to add new lines
if not os.path.exists(out_dir): os.makedirs(out_dir)
fout_name = out_dir + desc + '.dat'
fout = open(fout_name, 'w+')

# Copy configuration file used for this run to results directory
shutil.copyfile("TEA_config.py", out_dir + "TEA_config.py")

# Copy pre-atmosphere file used for this run to results directory
shutil.copyfile(infile, out_dir + infile.split("/")[-1][:-4] + "-preatm.dat")

# Read pre-atm file
fin  = open(infile, 'r+')
inlines = fin.readlines()
fin.close()

# From pre-atm file take all lines to marker[0]=end_head and write to atm file
# Those include header, species names, and labels for data
for i in np.arange(end_head):
    fout.writelines([l for l in inlines[i]])

# Write corrected species list into pre-atm file and continue
for i in np.arange(np.size(spec_list)):
    fout.write(spec_list[i] + ' ')
fout.write("\n\n")
fout.write("#FINDTEA\n")

# Write data from pre-atm file into each column of atm file 
fout.write(radi_arr[0].rjust(10) + ' ')
fout.write(pres_arr[0].rjust(10) + ' ')
fout.write(temp_arr[0].rjust(7) + ' ')
for i in np.arange(np.size(spec_list)):
    fout.write(spec_list[i].rjust(10)+' ')
fout.write('\n')

# Times / speed check for pre-loop runtime
if times:
    new = time.time()
    elapsed = new - end
    print("pre-loop:           " + str(elapsed))

# Detect operating system for sub-process support
if os.name == 'nt': inshell = True    # Windows
else:               inshell = False   # OSx / Linux

# Loop over all lines in pre-atm file and execute TEA loop
for q in np.arange(n_runs)[1:]:

    # Print for debugging purposes
    if doprint:
        print("\nReading atm file, TP line " + str(q))
    else:
        print('\n'+ str(q))
    
    # Radius, pressure, and temp for the current line    
    radi = radi_arr[q]
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
        = form.readoutput('results/' + desc + '/' + single_res[0])
    
    # Insert results from the current line (T-P) to atm file
    fout.write(radi.rjust(10) + ' ')
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
        old_name = loc_headerfile
        os.remove(old_name)
        shutil.rmtree("headers/" + desc + "/")
    
    # Save or delete lagrange.py, lambdacorr.py outputs and single T-P results
    if save_outputs:
        # Save directory for each T-P and its output files
        if not os.path.exists(loc_outputs): os.makedirs(loc_outputs)
        old_name = loc_outputs_temp
        new_name = loc_outputs + desc + "_" + '%.0f'%float(temp) + "K_"  \
                   '%1.2e'%float(pres) + "bar" + loc_outputs[-1::]
        if os.path.exists(new_name):
            for file in os.listdir(new_name):
                os.remove(new_name + file)
            shutil.rmtree(new_name)
        os.rename(old_name, new_name)
        shutil.rmtree(loc_transient)

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
        old_name = loc_outputs_temp
        for file in os.listdir(old_name):
            os.remove(old_name + file)
        shutil.rmtree(old_name)
        # Delete single T-P result directories and files
        for file in single_res:
            os.remove(out_dir + file)

# Close atm file
fout.close()

# Time / speed testing
if times:
    end = time.time()
    elapsed = end - start
    print("Overall run time:   " + str(elapsed))
