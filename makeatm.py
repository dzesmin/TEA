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
import os

# =============================================================================
# This code produces a pre-atm file in the format that TEA can read. It first 
# accepts user inputs of desired elemental and molecular species, as well as 
# ranges of radius, pressure, and temperature in the atmosphere. Then it reads
# elemental dex abundance data from an ASCII file described in the user manual 
# (default: abusndances.txt, Asplund et al. 2009), converts dex abundances to
# number density and divides them by the sum of all number densities in the
# mixture to get fractional abundances. Finally, it writes the header of the 
# pre-atm file, as well as the arrays for radius, pressure, temperature, and
# abundances for each layer in the atmosphere.
#
# User needs to edit the following lines:
# - line  72 with the name of the pre-atm file
# - line  73 with element names as they appear in the periodic table
# - line  74 with species names as readJANAF.py produces them. See
#            sorted version of the conversion-record.txt for the correct
#            names of the species. MUST include all elements and their states
# - line  76 with number of layers in the atmosphere
# - lines 77, 78 or 79 with radius and pressure values at the top and bottom of 
#            the atmosphere (linear or log range for pressure)
# - line  80 with temperatures at the top and bottom of the atmosphere
#            Note: the code currently works most precisely with hot-Jupiters'
#                  temperature range (1000 to 4000 K)
#
# Possible user errors in making pre-atm file that conflict with TEA:
#  - If H_g element is included H2_ref species must be included
#  - If the code stalls at the first iteration of the first temperature, check 
#    if all elements that appear in the in_elem list are also included in the
#    out_spec list with their correct names as readJANAF.py produces them.
#
# This code runs with the simple call: makeatm.py
#
# =============================================================================

# ============= START for user inputs =============== #

filename    = "Jupiter-log.dat"         # Output atm file name             
in_elem     = "H He C O N"          # List of input elements
out_spec    = "H_g He_ref C_g N_g O_g H2_ref CO_g CO2_g CH4_g H2O_g N2_ref NH3_g C2H2_g C2H4_g"
              # Output species, MUST INCLUDE elemental species and their states             
steps       = 100                             # Number of layers in the atmosphere
rad         = np.linspace( 1e4,    0, steps)  # Equispaced radius array
pres        = np.logspace(  -5,    2, steps)  # Equispaced pressure array
#pres        = np.linspace(  1,    1, steps)   # Equispaced pressure array
temp        = np.linspace( 100, 4000, steps)  # Equispaced temperature array
            
# =============  END for user inputs  =============== #

# Get inputs directory, create if non-existent
inputs_dir = "inputs/"
if not os.path.exists(inputs_dir): os.makedirs(inputs_dir)

# File containing solar abundances and weights
abun   = 'abundances.txt' 

# Read abundance data and convert to an array
f = open(abun, 'r')
abundata = []
for line in f.readlines():
    l = [value for value in line.split()]
    abundata.append(l)
abundata = np.asarray(abundata)
f.close()

# Trim abundata to elements we need
in_elem   = in_elem.split(" ")
nspec  = np.size(in_elem)
data_slice = np.zeros(abundata.shape[0], dtype=bool)
for i in np.arange(nspec):
    data_slice += (abundata[:,1] == in_elem[i])

# List of elements of interest and their corresponding data
abun_trim = abundata[data_slice]

# Take data and create list
out_elem = abun_trim[:,1].tolist()
out_dex  = abun_trim[:,2].tolist()

# Convert strings to floats
out_dex  = map(float, abun_trim[:,2])

# Convert dex exponents to elemental counts
out_num  = 10**np.array(out_dex)

# Get fractions of elemental counts to the 
#     total sum of elemental counts in the mixture
out_abn  = (out_num / np.sum(out_num)).tolist()

# Convert fractions to strings in scientific notation
for n in np.arange(np.size(out_abn)):
    out_abn[n] = str('%1.10e'%out_abn[n])

# Make a list of all data of interest
out = [['    Radius'] + ['Pressure'] + ['Temp'] + out_elem]

# Fill in data list
for i in np.arange(steps):
    out.append(['%8.3f'%rad[i]] + ['%8.4e'%pres[i]] + \
                          ['%7.2f'%temp[i]] + out_abn)

# Pre-atm header with basic instructions
header      = "# This is a TEA pre-atmosphere input file.            \n\
# TEA accepts a file in this format to produce species abundances as \n\
# a function of pressure and temperature.                            \n\
# Output species must be added in the line immediately following the \n\
# FINDSPEC marker and must be named to match JANAF converted names. "

# Place file into inputs directory 
inputs_out = inputs_dir + filename

# Write pre-atm file
f = open(inputs_out, 'w+')
f.write(header + '\n\n')
f.write('#FINDSPEC\n' + out_spec + '\n\n')
f.write('#FINDTEA\n')
for i in np.arange(steps + 1):
    # Radius list
    f.write(out[i][0].rjust(10) + ' ')
        
    # Pressure list
    f.write(out[i][1].rjust(10) + ' ')
    
    # Temp list
    f.write(out[i][2].rjust(7) + ' ')
    
    # Elemental abundance list
    for j in np.arange(nspec):
        f.write(out[i][j+3].rjust(16)+' ')
    f.write('\n')
f.close()

