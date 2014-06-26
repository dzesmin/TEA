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

import os
import shutil
import subprocess
import numpy as np
from sys import argv

import makeheader as mh

# =============================================================================
# This program runs TEA over an input file that contains only one T-P.
# The code retrieves the input file and the current directory name given by the 
# user. It sets locations of all necessary modules and directories that 
# will be used. Then, it executes the modules in the following order:
# makeheader.py, balance,py, and iterate.py. The final results with the input
# and the configuration files are saved in the results/ directory.
#
# This module prints on screen the code progress: the current T-P line from
# the pre-atm file, the current iteration number, and informs the user that
# minimization is done.
# Example:
#  100
# Maximum iteration reached, ending minimization.
#
# The program is executed with in-shell inputs:
# runsingle.py <input file> <name of the result directory>
# Example: runsingle.py inputs/Examples/inp_Example.txt Single_Example
# =============================================================================
    
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

# Retrieve user inputs file
infile  = argv[1:][0] 
desc    = argv[1:][1]

# Set up locations of necessary scripts and directories of files
cwd = os.getcwd() + '/'
thermo_dir     = cwd + "gdata"
loc_makeheader = cwd + "makeheader.py"
loc_balance    = cwd + "balance.py"
loc_iterate    = cwd + "iterate.py"
loc_headerfile   = cwd + "/headers/" + desc + "/header_" + desc + ".txt"
loc_outputs      = cwd + "/outputs/" + desc + "/"
loc_outputs_temp = cwd + "/outputs/" + "transient/" + desc + "/"
out_dir          = cwd + "/results/" + desc + "/"
single_res       = ["results-machine-read.txt", "results-visual.txt"]

# Create results directory
if not os.path.exists(out_dir): os.makedirs(out_dir)

# Copy configuration file used for this run to results directory
shutil.copyfile("TEA_config.py", out_dir + "TEA_config.py")

# Copy pre-atm file used for this run to results directory
shutil.copyfile(infile, out_dir + infile.split("/")[-1])

# Detect operating system for sub-process support
if os.name == 'nt': inshell = True    # Windows
else:               inshell = False   # OSx / Linux

# Execute main TEA loop
mh.make_singleheader(infile, desc, thermo_dir)
subprocess.call([loc_balance, loc_headerfile, desc, str(doprint)], shell=inshell)
subprocess.call([loc_iterate, loc_headerfile, desc, str(doprint)], shell=inshell)

# Save or delete lagrange.py and lambdacorr.py outputs
if save_outputs:
    old_name = loc_outputs_temp
    new_name = loc_outputs
    if os.path.exists(new_name): shutil.rmtree(new_name)
    os.rename(old_name, new_name)
else:
    shutil.rmtree(loc_outputs_temp)
