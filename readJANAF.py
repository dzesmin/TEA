#! /usr/bin/env python

# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
#
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Lead scientist and coder Jasmina Blecic, 
# assistant coder for the first pre-release Oliver M. Bowman.  
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

import numpy as np
from ast import literal_eval
from sys import argv
import re
import os
import shutil

from prepipe import setup

# =============================================================================
# This code makes the thermo_dir (default: 'gdata') directory that carries 
# converted JANAF tables with the information needed for TEA to run: 
# ['T (K)', '-[G-H(Tr)]/T (J/K/mol)', 'delta-f H (kJ/mol)']. 
#
# The code also makes conversion_record.txt that gives the names of the 
# original, raw JANAF files and the new names given by the readJANAF.py. 
# To sort the file list in alphabetical order, execute in terminal: 
# sort conversion_record.txt >conversion_record_sorted.txt
#
# The names of the files have the following format:
# 1. Each converted file carries a unique name and state given by the top
#    line in each JANAF file.
# 2. If a species appears just once in JANAF tables, it gets a unique name of
#    the compound and its state and is defined as 'originals' in the code 
#    (example: CH4_g).
# 3. If a species appears several times, it is defined as 'redundant' in the 
#    code and an additional string is added to differentiate among them. 
#    (example: Al2O3_cr_Alpha, Al2O3_cr_Beta, Al2O3_cr_Kappa).
# 4. If a species is an ion, additional string is added
#    (example: Al-007.txt and Al-008.txt became Al_ion_n_g, Al_ion_p_g)
#
# The setup of this code is made in prepipe.py inside the setup() function.
# The code retrieves pre-pipeline setup information, creates a directory for 
# converted thermodynamic files, checks whether a species is redundant in 
# JANAF tables, and creates conversion_record.txt. If a species is redundant,
# an additional string is added to the file name. The module loops over all 
# JANAF tables and writes data into the correct columns with the correct 
# labels.
#
# The prepipe.py code can execute this code together with the readJANAF.py. 
# readJANAF.py can also be executed on its own with the simple command: 
# readJANAF.py. 
# =============================================================================

# Enable various debug printouts, False by default
try:
    doprint = literal_eval(argv[1:][0])
except IndexError:
    doprint = False

# Print indicator that readJANAF.py is executing
print("\nRunning readJANAF...\n")

# Retrieve pre-pipeline setup information
raw_dir, thermo_dir, stoich_dir, stoich_out, abun_file, \
n_ele, n_JANAF, species, JANAF_files = setup()

# Create directory for converted thermodynamic files, 
#        else clear previously existing directory
if os.path.exists(thermo_dir): 
    shutil.rmtree(thermo_dir)
    os.makedirs(thermo_dir)
    if doprint:
        print("Thermodynamic directory already exists. \
                               Clearing old data...\n")
else: 
    os.makedirs(thermo_dir)
    if doprint:
        print("No thermodynamic directory found, creating new...\n")
        
# Establish column labels used for JANAF tables
header = ['T (K)', '-[G-H(Tr)]/T (J/K/mol)', 'delta-f H (kJ/mol)']
 
# Create conversion record
file_list = open('conversion_record.txt', 'w+')

# Allocate dictionaries for original/redundant check
originals = {}
redundant = {} 

# Check for redundancies, only insert additional label if more than one
#       species with same formula is in data
for i in np.arange(n_JANAF):
    infile = raw_dir + JANAF_files[i]
    outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) + '.txt'
    if originals.has_key(outfile):
        redundant[outfile] = infile
    else:
        originals[outfile] = infile
    
# Set initial number of files read and label boolean for ignored files
n_files = 0
print_label  = True

# Loop over all JANAF tables
for i in np.arange(n_JANAF):
    # Read-in current JANAF table
    infile = raw_dir + JANAF_files[i]
    f = open(infile, 'r')
    
    # Read-in data from JANAF table
    data = []
    getpressure = True
    for line in f.readlines():
        l = [value for value in line.split()]
        # Find line containing pressure
        if getpressure:
            pressline = ' '.join(l) 
            getpressure = False 
        haschar = re.findall('[A-z]', ' '.join(l))
        hasnum = re.findall('[0-9]', ' '.join(l))
        length = np.size(l)
        
        # Ignore lines with comments or with missing data
        if (haschar == [] and length >= 8 and hasnum != []):
            data.append(l)
    f.close()
    
    # Get pressure from JANAF table
    if re.search(', (.*?) Bar \(', pressline):
        # Removes ' Bar' from the pressure string
        pressure = np.int(re.search(', (.*?) Bar \(', pressline).group(1))
    else:
        # If pressure is not listed, JANAF table assumes 1 bar
        pressure = 1 
    
    # Put data into array with correct data types and labels
    n_temps = np.shape(data)[0]
    gdata = np.empty((n_temps, 3), dtype='|S50')
    for j in np.arange(n_temps):
        gdata[j,0] = data[j][0]
        gdata[j,1] = data[j][3]
        gdata[j,2] = data[j][5]
    
    # Create file for each JANAF species, adding additional string if redundant
    # For consistency, only use 1 bar tables (only H2O has other pressures)
    if pressure == 1: 
        outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) +   \
                                                                   '.txt'
        # Check if species is redundant and add additional string if so
        if redundant.has_key(outfile):
            outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) \
                                        + '_' + np.str(species[i,3]) + '.txt'
        elif os.path.isfile(outfile):
            outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) \
                                        + '_' + np.str(species[i,3]) + '.txt'
        
        # Write conversion record on-screen if debugging is allowed
        if doprint:
            print(outfile.split('.')[0] + ' made from ' + infile)

        # Write conversion data to file
        file_list.write(outfile+' made from ' + infile + '\n')

        # Write converted JANAF tables
        f = open(outfile, 'w+')
        n_files += 1

        # Write column labels
        f.write(header[0].ljust(8))
        f.write(header[1].ljust(24))
        f.write(header[2].ljust(22))
        f.write('\n')

        # Write data in corresponding columns
        for h in np.arange(n_temps):
            f.write(gdata[h, 0].ljust(8))
            f.write(gdata[h, 1].ljust(24))
            f.write(gdata[h, 2].ljust(22))
            f.write('\n')
        
        f.close()

    # Exclude files with pressure != 1 bar
    else:
        if print_label:
            print('These files were not included because pressure != 1bar:')
            print_label = False
        print('    ' + infile.split('/')[-1] + ' (' + species[i, 0] + '_'     \
               + np.str(species[i,2]) + ')')

# Print directory names to screen
print("\nSaved " + str(n_files) + " TEA-read files to \'" + thermo_dir +      \
       "\', out of the available \n      " + str(n_JANAF) +                    \
       " JANAF files in the \'" + raw_dir + "\' directory.")

# Print conversion record file name to screen
print("\nSaved list of converted files to \'conversion_record.txt.\'")
file_list.close()

