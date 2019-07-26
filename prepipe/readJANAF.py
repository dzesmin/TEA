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
from ast import literal_eval
from sys import argv
import re
import os
import shutil

from prepipe import *


location_TEA = os.path.realpath(os.path.dirname(__file__) + "/..") + "/"

# =============================================================================
# This code makes the thermo_dir (default: 'gdata') directory that carries
# converted JANAF tables with the information needed for TEA to run:
# ['T (K)', '-[G-H(Tr)]/T (J/K/mol)', 'delta-f H (kJ/mol)'].
#
# The code also makes conversion_record.txt that gives the names of the
# original, the raw JANAF files and the new names given by the readJANAF.py.
# To sort the file list in alphabetical order, execute in terminal:
# sort conversion_record.txt >conversion_record_sorted.txt
#
# The names of the files have the following format:
# 1. Each converted file carries a unique name and state given by the top
#    line in each JANAF file.
# 2. If a species appears just once in the JANAF tables, it gets a unique name
#    of the compound and its state and is defined as 'originals' in the code
#    (example: CH4_g).
# 3. If a species appears several times, it is defined as 'redundant' in the
#    code and an additional string is added to differentiate among them.
#    (example: Al2O3_cr_Alpha, Al2O3_cr_Beta, Al2O3_cr_Kappa).
# 4. If a species is an ion, additional string is added
#    (example: Al-007.txt and Al-008.txt became Al_ion_n_g, Al_ion_p_g)
#
# The setup of this code is made in prepipe.py inside the setup() function.
# The code retrieves pre-pipeline setup information, creates a directory for
# converted thermodynamic files, checks whether a species is redundant in the
# JANAF tables, and creates conversion_record.txt. If a species is redundant,
# an additional string is added to the file name. The module loops over all
# JANAF tables and writes data into the correct columns with the correct
# labels.
#
# The prepipe.py code can execute this code together with the readJANAF.py.
# readJANAF.py can also be executed on its own with the simple command:
# readJANAF.py.
# =============================================================================

# Print indicator that readJANAF.py is executing
print("\nRunning readJANAF...\n")

# Retrieve pre-pipeline setup information
raw_dir, thermo_dir, stoich_dir, stoich_out, abun_file, \
  n_ele, n_JANAF, species, JANAF_files = setup(abun_file)

# Create directory for converted thermodynamic files,
#        else clear previously existing directory
if os.path.exists(thermo_dir):
    shutil.rmtree(thermo_dir)
    os.makedirs(thermo_dir)
    if verb > 1:
        print("Thermodynamic directory already exists.\n"
               "Clearing old data...\n")
else:
    os.makedirs(thermo_dir)
    if verb > 1:
        print("No thermodynamic directory found, creating new, lib/gdata/.\n")

# Establish column labels used for JANAF tables
header = ['T (K)', '-[G-H(Tr)]/T (J/K/mol)', 'delta-f H (kJ/mol)']

# Create conversion record
file_list = open(location_TEA + 'lib/conversion_record.txt', 'w+')

# Allocate dictionaries for original/redundant check
originals = {}
redundant = {}

# Check for redundancies, only insert additional label if more than one
#       species with same formula is in data
for i in np.arange(n_JANAF):
    infile = raw_dir + JANAF_files[i]
    print("thermo_dir", thermo_dir, species[i, 0], np.str(species[i,2]))
    outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) + '.txt'
    if originals.__contains__(outfile):
        redundant[outfile] = infile
    else:
        originals[outfile] = infile

# Set initial number of files read and label boolean for ignored files
n_files = 0
print_label = True

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
    gdata = np.empty((n_temps, 3), dtype='U100')
    for j in np.arange(n_temps):
        gdata[j,0] = data[j][0]
        gdata[j,1] = data[j][3]
        gdata[j,2] = data[j][5]

    # Create file for each JANAF species, adding additional string if redundant
    # For consistency, only use 1 bar tables (only H2O has other pressures)
    if pressure == 1:
        # Catch weird species
        if '.' in species[i, 1]:
            pass
        else:
            outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) +   \
                                                                   '.txt'
            # Renaming case-insensitive species names for MAC users
            MACspecies = species[i, 0] + '_' + np.str(species[i,2])
            if MACspecies=='CoCl2_g'  or MACspecies=='CoCl_g' or \
                MACspecies=='CoF2_g'  or MACspecies=='Co_g'   or \
                MACspecies=='Cs2_g'   or MACspecies=='Cs_g'   or \
                MACspecies=='Hf_g':
                outfile = outfile[:-5] + 'gas.txt'

            # Check if species is redundant and add additional string if so
            elif redundant.__contains__(outfile):
                outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) \
                                        + '_' + np.str(species[i,3]) + '.txt'
            elif os.path.isfile(outfile):
                outfile = thermo_dir + species[i, 0] + '_' + np.str(species[i,2]) \
                                        + '_' + np.str(species[i,3]) + '.txt'

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
print("\nTo sort the file alphabetically do: \
        \nsort conversion_record.txt >conversion_record_sorted.txt\n")

file_list.close()


