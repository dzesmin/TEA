#! /usr/bin/env python

# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
#
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Lead scientist and coder Jasmina Blecic, 
# assistant coder Oliver M. Bowman.  
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

from readconf import *

import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# =============================================================================
# This code produces a pre-atmospheric file in the format that TEA can read it.
# The file will be placed in the inputs/pre_atm/ directory. The module 
# consists of 2 functions:
# readPT()   reads pressure-temperature(PT) profile from the PT file provided
# makeeatm() writes a pre-atm file
#
# The TEA.cfg file should be edit with the following information: 
# pre-atmospheric file name, input elemental species, and output species.
#
# Run the code as: makeatm.py <config file name> <PT file name>
#
# Possible user errors in configuring pre-atm section in the TEA.cfg that
# conflicts with TEA:
#  - in input_elem use elements names as they appear in the periodic table
#  - input_elem MUST be listed in the order as they appear in the 
#   'abundances.txt' file provided
#  - H, and He elements as input_elem must be included for hot-Jupiters
#  - H_g, He_ref and H2_ref species in output_species must be included for
#    hot-Jupiters
#  - all input_elem must be included in the list of output_species with their 
#    states. They MUST be listed in the same order as they appear in the
#   'abundances.txt' file
#  - use species names as readJANAF.py produces them. See sorted version of the
#    conversion-record.txt for the correct names of the species. 
#  - If the code stalls at the first iteration of the first temperature, check 
#    if all elements that appear in the species list are included with their 
#    correct names.
# =============================================================================



# =============== READ PT FILE ================== 
def readPT(PT_file):
    """
    Read the PT file contaitning pressure and temperature arrays.

   MUST BE IN THIS FORMAT IF USER_MADE PT FILE
    Parameters:
    abun_file: string
             Name of the abundances file (default:abundances.txt)

    Returns:
    abun_trim: list of strings

    Notes: in_elem MUST be in the order thay appear in 'abundances.txt'

    Revisions
    ---------
    2014-09-10  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    """

    # read abundance data and convert to array:
    f = open(PT_file, 'r')
    data = []
    for line in f.readlines():
        if line.startswith('#'):
            continue
        else:
            l = [value for value in line.split()]
            data.append(l)
    data = np.asarray(data)
    f.close()

    # size of the data array (number of layers in the atmopshere)
    ndata = len(data)

    # allocates arrays of pressure and temperature
    pressure    = []
    temperature = []

    # reads lines and store pressure and temperature data
    for i in np.arange(ndata):
        pressure    =  np.append(pressure, data[i][0])
        temperature =  np.append(temperature, data[i][1])
    pres = pressure.astype(float)
    temp = temperature.astype(float)

    return pres, temp

def makeatm():
    '''
    This code produces a pre-atm file in the format that TEA can read it. It  
    defines the directory where the pre-atm file will be placed, then it reads
    the configuration file and elemental dex abundance data file 
    (default: abundances.txt). It trims the abundance data to the elements of 
    interest, and takes the column with the weights information to calculate 
    the mean molecular weight, based on an assumption that 85% of the atmosphere 
    is filled with H2 and 15% with He. Then, it converts dex abundances of all 
    elements of interest to number density and divides them by the sum of all 
    number densities in the mixture to get fractional abundances.
    It calls the radpress() function to calculate radii, and writes
    the data (radii, pressure, temperature, elemental abundances) into a pre-atm
    file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

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

    # Get pre_atm directory
    pre_atm_dir = "inputs/pre_atm/"
    if not os.path.exists(pre_atm_dir): os.makedirs(pre_atm_dir)
 
    # Get the name of the configuration file
    conf_name = sys.argv[1]  

    # Get the name of the pressure file
    PT_file = sys.argv[2] 

    # Read pressure and temperature data
    pressure, temp = readPT(PT_file)

    # Read abundance data and convert to an array
    f = open('abundances.txt', 'r')
    abundata = []
    for line in f.readlines():
        if line.startswith('#'):
            continue
        else:
            l = [value for value in line.split()]
            abundata.append(l)
    abundata = np.asarray(abundata)
    f.close()

    # Trim abundata to elements we need
    in_elem_split = input_elem.split(" ")
    nelem  = np.size(in_elem_split)
    data_slice = np.zeros(abundata.shape[0], dtype=bool)
    for i in np.arange(nelem):
        data_slice += (abundata[:,1] == in_elem_split[i])

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

    # Make a list of labels
    out = ['#Pressure'.ljust(10)] + [' Temp'.ljust(7)]
    for i in np.arange(nelem):
         out = out + [out_elem[i].ljust(16)]
    out = [out]

    # Number of layers in the atmosphere
    n_layers = len(pressure)

    # Fill in data list
    for i in np.arange(n_layers):
        out.append(['%8.4e'%pressure[i]] + \
                          ['%7.2f'%temp[i]] + out_abn)

    # Pre-atm header with basic instructions
    header      = "# This is a TEA pre-atmosphere input file.            \n\
# TEA accepts a file in this format to produce species abundances as \n\
# a function of pressure and temperature.                            \n\
# Output species must be added in the line immediately following the \n\
# #SPECIES marker and must be named to match JANAF converted names.  \n\
# Units: pressure (bar), temperature (K), abundance (unitless)."

    # Write pre-atm file
    f = open(pre_atm_dir + pre_atm_name, 'w+')
    f.write(header + '\n\n')
    f.write('#SPECIES\n' + output_species + '\n\n')
    f.write('#TEADATA\n')
    for i in np.arange(n_layers + 1):
    
        # Pressure list
        f.write(out[i][0].ljust(10) + ' ')
    
        # Temp list
        f.write(out[i][1].ljust(7) + ' ')
    
        # Elemental abundance list
        for j in np.arange(nelem):
            f.write(out[i][j+2].ljust(16)+' ')
        f.write('\n')
    f.close()
   
    print "\nPre-atmospheric file is created and placed in the inputs/pre_atm/ folder.\n"


if __name__ == "__main__":
   makeatm()
