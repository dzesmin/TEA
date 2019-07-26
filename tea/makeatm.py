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
import ntpath
import os
import shutil
import matplotlib.pyplot as plt
import sys
import six

import readconf as rc


# =============================================================================
# This module produces a pre-atmospheric file in the format that TEA can read it.
# The pre-atmospheric file will be placed in the atm_inputs/ directory.
# The module consists of 2 functions:
# readPT()   reads pressure-temperature (PT) profile from the PT file provided
# makeatm()  writes a pre-atm file
#
# The TEA.cfg file need to be edit with the following information: PT file,
# pre-atmospheric file name, input elemental species, and output species.
#
# Run the code as: makeatm.py <DIRECTORY_NAME>
#
# Possible user errors in configuring pre-atm section in the TEA.cfg that
# conflicts with TEA:
#  - in input_elem use elements' names as they appear in the periodic table
#  - all input_elem must be included in the list of output_species with their
#    states at the begining of the output_species list.
#  - use species names as readJANAF.py produces them. See lib/gdata folder or
#    lib/conversion_record_sorted.txt for the correct names of the species
#  - H, and He elements as input_elem must be included for hot-Jupiters
#  - H_g, He_ref and H2_ref species in output_species must be included for
#    hot-Jupiters
#  - TEA does not work with ionized and condensate species
# =============================================================================

# Read configuration-file parameters
TEApars, PREATpars = rc.readcfg()
maxiter, savefiles, verb, times, abun_file, location_out, xtol, ncpu = TEApars
PT_file, pre_atm_name, input_elem, output_species = PREATpars

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
if location_out[-1] != '/':
    location_out += '/'

# Retrieve user directory name
desc  = sys.argv[1:][0]

if verb==2:
    # Check if output directory exists and inform user
    if os.path.exists(location_out + desc):
        six.moves.input("  Output directory " + str(location_out + desc) + 
                        "/\n  already exists.\n"
                        "  Press enter to continue and overwrite existing files,\n"
                        "  or quit and choose another output name.\n")

# Create inputs directory
inputs_dir = location_out + desc + "/atm_inputs/"
if not os.path.exists(inputs_dir): os.makedirs(inputs_dir)


def readPT(PT_file):
    """
    Reads a PT file containing pressure and temperature arrays. 
    If custom made must be in the format provided in doc/examples/ folder.

    Parameters
    ----------
    PT_file: string
          Path to the temperature and pressure file.

    Returns
    -------
    pres: 1D array of floats
          Array containing pressures in each layer.
    temp: 1D array of floats
          Array containing temperatures in each layer.
    """

    # Read abundance data and convert to array
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

    # Size of the data array (number of layers in the atmosphere)
    ndata = len(data)

    # Allocate arrays of pressure and temperature
    pressure    = []
    temperature = []

    # Read lines and store pressure and temperature data
    for i in np.arange(ndata):
        pressure    = np.append(pressure, data[i][0])
        temperature = np.append(temperature, data[i][1])
    pres = pressure.astype(float)
    temp = temperature.astype(float)

    return pres, temp


def makeatm():
    '''
    Produces a pre-atmospheric file in the format that TEA can
    reads it. The file will be placed in atm_inputs/ directory. It calls
    readPT() function to take pressure and temperature array and reads the
    elemental abundance data file (default: abundances.txt,
    Asplund et al. 2009). The code trims the abundance data to the elements
    of interest, converts species dex abundances (logarithmic abundances,
    dex stands for decimal exponent) into number densities and divides them
    by the hydrogen number densities fractional abundances. It writes data
    (pressure, temperature, elemental abundances) into a pre-atmospheric
    file. The config file, pressure and temperature file, and the abundances
    file are copied to the atm_inputs/ directory.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Check if config file exists
    TEA_config = 'TEA.cfg'
    try:
        f = open(TEA_config)
    except IOError:
        print("\nMissing config file, place TEA.cfg in the working directory.\n")

    # Inform user if TEA.cfg file already exists in inputs/ directory
    if os.path.isfile(inputs_dir + TEA_config):
        print("  {:s} overwritten in atm_inputs/ dir.".format(TEA_config))
    # Copy TEA.cfg file to current inputs directory
    shutil.copy2(TEA_config, inputs_dir + TEA_config)

    # Inform user if PT file already exists in inputs/ directory
    head, PT_filename   = ntpath.split(PT_file)
    if os.path.isfile(inputs_dir + PT_filename):
        print("  PT file, {:s}, overwritten in atm_inputs/ dir.".format(PT_filename))
    # Copy PT file to current inputs directory
    shutil.copy2(PT_file, inputs_dir + PT_filename)

    # Inform user if abundances file already exists in inputs/ directory
    head, abun_filename = ntpath.split(abun_file)
    if os.path.isfile(inputs_dir + abun_filename):
        print("  {:s} overwritten in atm_inputs/ dir.".format(abun_filename))
    # Copy abundances file to inputs/ directory
    shutil.copy2(abun_file, inputs_dir + abun_filename)

    # Inform user if pre-atm file already exists in inputs/ directory
    if os.path.isfile(inputs_dir + pre_atm_name):
        print("  pre_atm file, {:s}, overwritten in atm_inputs/ dir.".format(pre_atm_name))

    # Read pressure and temperature data
    pressure, temp = readPT(PT_file)

    # Read abundance data and convert to an array
    f = open(abun_file, 'r')
    abundata = []
    for line in f.readlines():
        if line.startswith('#'):
            continue
        else:
            l = [value for value in line.split()]
            abundata.append(l)
    abundata = np.asarray(abundata)
    f.close()

    # Get input elements
    in_elem_split = input_elem.split(" ")
    nelem  = np.size(in_elem_split)

    # Get output elements in output species
    out_elem = []
    out_spec = output_species.split(" ")
    for i in np.arange(len(out_spec)):
        out_spec[i] = out_spec[i].partition('_')[0]
        if len(out_spec[i]) == 1:
            out_elem.append(out_spec[i])
        elif len(out_spec[i]) == 2 and out_spec[i][1].islower():
            out_elem.append(out_spec[i])

    # Catch if input elements not included in output species
    for i in in_elem_split:
        if not any(i in j for j in out_elem):
            raise IOError(
                    "\n\nSome input elements not included in output species."
                      "\nCorrect TEA.cfg and rerun.\n")

    # Trim abundata to elements we need
    data_slice = np.zeros(abundata.shape[0], dtype=bool)
    for i in np.arange(nelem):
        data_slice += (abundata[:,1] == in_elem_split[i])

    # List of elements of interest and their corresponding data
    abun_trim = abundata[data_slice]

    # Take data and create list
    out_elem = abun_trim[:,1].tolist()
    out_dex  = abun_trim[:,2].tolist()

    # Convert strings to floats
    out_dex  = list(map(float, abun_trim[:,2]))

    # Convert logarithmic (dex) exponents to number density
    out_num  = 10**np.array(out_dex)

    # Get hydrogen number density
    H_num = 10**12

    # Get fractions of element number density to hydrogen number density
    out_abn  = (out_num / H_num).tolist()

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
        out.append(['%8.4e'%pressure[i]] + ['%7.2f'%temp[i]] + out_abn)

    # Pre-atm header with basic instructions
    header = ("# This is a TEA pre-atmosphere input file.\n"
    "# TEA accepts a file in this format to produce species abundances as\n"
    "# a function of pressure and temperature.\n"
    "# Output species must be added in the line immediately following the\n"
    "# #SPECIES marker and must be named to match JANAF converted names.\n"
    "# Units: pressure (bar), temperature (K), abundance (unitless).")

    # Write pre-atm file
    f = open(inputs_dir + pre_atm_name, 'w')
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

    print("\nCreated pre-atmospheric file:\n"
            + str(inputs_dir + pre_atm_name) + "\n")


if __name__ == "__main__":
   makeatm()
