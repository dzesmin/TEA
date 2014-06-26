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

import numpy as np
from ast import literal_eval
from sys import argv
import os
import shutil

from prepipe import setup, comp

# =============================================================================
# This code makes the stoich_out file (default: stoich.txt) that carries
# stoichiometric values of all species that appear in the JANAF tables. It
# creates a temporary directory where JANAF tables are converted into
# stoichiometric tables. They carry a unique  name and state, given by the top
# line in each JANAF table:
# (i.e., original JANAF table "Al-001.txt" is converted to "Al_ref.txt").  
# If desired, user can  preserve this directory (in TEA_config.py, 
# doprint = True). 

# The names of the files in the temporary directory have the following format:
# 1. If a species appears just once in JANAF tables, it gets a unique name of
#    the compound and its state and is defined as 'originals' in the code 
#    (example: CH4_g).
# 2. If a species appears several times, it is defined as 'redundant' in the 
#    code and an additional string is added to differentiate among them. 
#    (example: Al2O3_cr_Alpha, Al2O3_cr_Beta, Al2O3_cr_Kappa).

# The setup of this code is made in prepipe.py inside setup().
# The code retrieves the setup information, creates a temporary directory to 
# store the converted files, allocates space for these files with a unique, 
# original names and files with redundant names, then loops over all JANAF 
# tables to write stoichiometric files. 

# To write the stoich_out file (default: stoich.txt), it again loops over all
# elements (listed in comp() inside prepipe.py) and matches them with the
# abundance data from abun_file (default: abundances.txt). If a match is not
# found, the abundance is set to zero. This information is written at the top 
# of the stoich_out file. 

# For each species in the JANAF tables and elements in comp(), the species name
# and stoichiometric values are written into the stoich_out file. The code
# ignores redundant compounds as they carry the same stoichiometric values.

# The prepipe.py code can execute the this code together with the readJANAF.py. 
# makestoich.py can also be executed on its own with the simple command: 
# makestoich.py. 
# =============================================================================

# Enable various debug printouts, False by default
try:
    doprint = literal_eval(argv[1:][0])
except IndexError:
    doprint = False

# Print indicator that makestoich.py is executing
print("\nRunning makestoich...\n")

# Retrieve pre-pipeline setup information
raw_dir, thermo_dir, stoich_dir, stoich_out, abun_file, \
n_ele, n_JANAF, species, JANAF_files = setup()


# ============= Create stoichiometry files directory =================== #

# Create directory for stoichiometric files, 
#        else clear previously existing directory
if os.path.exists(stoich_dir): 
    shutil.rmtree(stoich_dir)
    os.makedirs(stoich_dir)
    if doprint:
        print("Stoichiometric directory already exists. \
                                Clearing old data...\n")
else: 
    os.makedirs(stoich_dir)
    if doprint:
        print("No stoichiometric directory found, creating new...\n")

# Allocate dictionary for original check
originals    = {} 

# Loop over all JANAF files to write new stoichiometric files
n_files      = 0
for i in np.arange(n_JANAF):
    # Read list of elements from comp()
    elements = comp(species[i, 1])
    
    # Create stoichiometric file name for species with its state
    outfile = stoich_dir + species[i, 0] + '_' + species[i,2] + '.txt'

    # Check if isomer exists for specie, and if so add additional string
    if originals.has_key(outfile):
        outfile = stoich_dir + species[i, 0] + '_' + np.str(species[i,2]) + \
                                      '_' + np.str(species[i,3]) + '.txt'
    elif os.path.isfile(outfile):
        outfile = stoich_dir + species[i, 0] + '_' + np.str(species[i,2]) + \
                                      '_' + np.str(species[i,3]) + '.txt'
            
    # Open each file to write
    f = open(outfile, 'w+')
    numcols = np.shape(elements)[1]
    numele = np.shape(elements)[0]
    for j in np.arange(numele):
        for h in np.arange(numcols):
            f.write(np.str(elements[j, h]).ljust(4))
        f.write('\n')
    f.close()


# ============= Create all-in-one stoichiometry file =================== #
#              with abundances and all species/elements

# Retrieve abundance info
f = open(abun_file, 'r')
abundata = []

# Read in data from abundance file
for line in f.readlines():
    l = [value for value in line.split()]
    abundata.append(l)

# Place data in array
abundata = np.asarray(abundata)
f.close()

# Start writing stoichiometry file
f = open(stoich_out, 'w+')
f.write('b'.ljust(12))

# Place abundance data for each element into stoichiometry file 
for i in np.arange(n_ele):
    # Get elements listing
    element = elements[i, 1]

    # Check if specific element's abundances is non-zero
    index = np.where(abundata[:, 1] == element)[0]

    # Elemental abundance is not in abundance file, assume 0.
    if index.size == 0:
        f.write('0.'.rjust(6))
    # Write elemental abundance into stoichiometry file
    else:
        ind = index[0]
        f.write(abundata[ind,2].rjust(6))

f.write('\n')

# Place row for elemental symbols into stoichiometry file
f.write('Species'.ljust(12))
for i in np.arange(n_ele):
    f.write(elements[i, 1].rjust(6))
f.write('\n')
 
# Allocate dictionaries for original/redundant check
originals = {}
redundant = {} 

# Read over all species and place elemental counts into stoichiometry file
for j in np.arange(n_JANAF):
    # Get name and elemental counts for specie
    specie = species[j, 0]
    elements = comp(species[j, 1])
    
    # Check for redundant chemical formulas for species. Only keep one of such
    #       formula in stoichiometry file due to identical elemental counts.
    if originals.has_key(specie):
        # Specie chemical formula is redundant; do nothing
        redundant[specie] = True
    else:
        # Species chemical formula is not redundant; place counts into file
        originals[specie] = False
        f.write(specie.ljust(12))
        
        # Write count for each element into file
        for i in np.arange(n_ele):
            f.write(np.str(elements[i, 2]).rjust(6))        
        f.write('\n')
f.close()

# Delete temporary stoichiometry directory as all information is now 
#        in one stoichiometric data file, stoich_out
if not doprint:
    shutil.rmtree(stoich_dir)
else:
    print("Conserving stoichiometry directory for individual species.")

