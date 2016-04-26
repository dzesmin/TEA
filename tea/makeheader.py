
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

import numpy as np
import re
import os

from scipy.interpolate import UnivariateSpline

# =============================================================================
# This module contains functions to write headers containing all necessary 
# chemical information for a single T-P and multiple
# T-P runs. It consists of two main functions, make_singleheader() and 
# make_atmheader() called by the runsingle.py and runatm.py modules
# respectively. The header_setup(), atm_headarr(), single_headarr(), and
# write_header() are the supporting functions for the main functions. 
# Imported by runatm.py and runsingle.py to create the header files.
# =============================================================================

# Correct directory names
if location_TEA[-1] != '/':
    location_TEA += '/'

if location_out[-1] != '/':
    location_out += '/'

def header_setup(temp, pressure, spec_list,                      \
                 thermo_dir, stoich_file = 'lib/stoich.txt'):
    '''
    This function is a common setup for both single T-P and multiple T-P runs. 
    Given the thermochemical data and stoichiometric table, this function 
    returns stoichiometric values and an array of chemical potentials for the 
    species of interest at the current temperature and pressure.  It also 
    returns an array of booleans that marks which information should be read 
    from stoich_file for the current species. It is executed by the 
    make_atmheader() and make_singleheader() functions.

    Parameters
    ----------
    temp: float
         Current temperature value.
    pressure: float
         Current pressure value.
    spec_list: string array
         Array containing names of molecular species.
    thermo_dir = 'lib/gdata':  string
         Name of directory containing thermodynamic data.
    stoich_file = 'lib/stoich.txt': string
         Name of file containing stoichiometric data.

    Returns
    -------
    stoich_data: array
         Full stoichiometric information from stoich_file for species used.
    spec_stoich: float array
         Array containing values from stoich_file that correspond to the 
         species used.
    g_RT: float array
         Array containing species' chemical potentials.
    is_used: boolean array
         Array containing booleans to trim stoichiometric data to only the 
         species of interest.
    '''

    # Ensure that inputs are floats
    temp     = np.float(temp)
    pressure = np.float(pressure)

    # Obtain thermo_dir files, and count both files and species
    gdata_files = os.listdir(thermo_dir)
    n_gdata     = np.size(gdata_files)
    n_spec      = np.size(spec_list)

    # Create index of where species listed in the input file are in thermo_dir
    spec_ind = np.zeros(n_spec)
    for i in np.arange(n_spec):
        spec_file = spec_list[i] + '.txt'
        if spec_file in gdata_files:
            spec_ind[i] = gdata_files.index(spec_file)

    # Create array of G/RT
    g_RT = np.zeros(n_spec)
    R = 8.3144621 # J/K/mol
    for i in np.arange(n_spec):
        spec_file = thermo_dir + '/' + gdata_files[np.int(spec_ind[i])]
        f = open(spec_file, 'r')
        data = []
        headerline = True
        for line in f.readlines():
            if headerline:
                headerline = False
            else:
                l = [np.float(value) for value in line.split()]
                data.append(l)
        
        f.close()
       
        # Convert data to an array
        data = np.asarray(data)

        # Equation for g_RT term equation (10) in TEA theory document
        #  G        G-H(298)      delta-f H(298)
        # ---  =    -------  +    -------------
        # R*T         R*T              R*T

        # First term is divided by T in JANAF, equation (11) in TEA theory document   
        # Second term needs to be converted to Joules (JANAF gives kJ)
        #  G     G-H(298)             1                                    1000
        # ---  = ------- [J/K/mol] * ---         + delta-f H(298) [kJ/mol] ------ 
        # R*T      T                  R [J/K/mol]                          R*T [J/K/mol][K] 

        # Spline interpolation of JANAF term1 values
        spline_term1 = UnivariateSpline(data[:, 0], data[:,1], s=1)
        gdata_term1  = spline_term1(temp)
       
        # Take term2 from gdata/ tables
        for j in np.arange(len(data[:,0])):
            if data[:,0][j] == 298.15:
                gdata_term2  = data[:,2][j]   

        # Calculate the above equation   
        g_RT[i] = - (gdata_term1 / R) + (gdata_term2 * 1000 / (temp * R))
        
    # Get number of elements that occur in species of interest
    nostate = np.copy(spec_list)
    for i in np.arange(n_spec):
        nostate[i] = re.search('(.*?)_', spec_list[i]).group(1)

    # Location of the stoich_file
    stoich_file = location_TEA + stoich_file

    # Get stoichiometric information for species of interest
    f = open(stoich_file, 'r')
    stoich_data = []
    bline = True
    for line in f.readlines():
        l = [value for value in line.split()]
        stoich_data.append(l)

    # Store information in stoich_data array
    stoich_data = np.asarray(stoich_data)
    f.close()
    
    # Count total number of elements in stoich_file
    n_ele = np.size(stoich_data[0, 1:])
    
    # Allocate array to store current element and species stoichiometric values
    spec_stoich = np.empty((n_spec+1, n_ele), dtype=np.float)
    spec_stoich[0] = stoich_data[0,1:]

    # Place species' stoichiometric data into array
    for i in np.arange(n_spec):
        idx = np.where(stoich_data[:, 0] == nostate[i])[0]
        if np.size(idx) != 1:
            idx = idx[0]
        spec_stoich[i+1] = stoich_data[idx,1:]
    
    # Determine which elements are used to trim down final stoichiometric table
    is_used = np.empty(n_ele, dtype=np.bool)
    for j in np.arange(n_ele):
        if np.sum(spec_stoich[1:, j]) != 0:
            is_used[j] = True
        else:
            is_used[j] = False

    return stoich_data, spec_stoich, g_RT, is_used


def single_headarr(spec_list, stoich_data, spec_stoich, is_used):
    '''
    This function gathers data needed for TEA to run in a single T-P case. 
    These are: elemental abundances, species names, and their stoichiometric
    values. For the list of elements and species used, it takes the abundances
    and stoichiometric values and puts them in the final array. This function
    is run by make_singleheader() and is dependent on results from 
    header_setup().

    Parameters
    ----------
    spec_list: string array
         Array containing names of molecular species.
    stoich_data: array
         Full stoichiometric information from header_setup() for species of 
         interest.
    spec_stoich: float array
         Array containing values from header_setup() for species of interest.
    is_used: boolean array
         Array containing booleans to trim stoichiometric data to only the 
         species of interest.

    Returns
    -------
    stoich_arr: array
         Array containing elemental abundances, species names, and their 
         stoichiometric values.
    '''
    
    # Get number of species used
    n_spec = np.size(spec_list)
    
    # Allocate final header array
    stoich_arr = np.empty((n_spec + 2, np.sum(is_used) + 1), dtype=np.object)
    
    # First row is 'b' values (elemental abundances)
    stoich_arr[0,0] = 'b'
    stoich_arr[0,1:] = stoich_data[0, np.where(is_used)[0] + 1]
    
    # Second row is list of species
    stoich_arr[1,0] = 'Species'
    stoich_arr[1,1:] = stoich_data[1, np.where(is_used)[0] + 1]
    
    # Each row after contains the weights of each element referred to as
    #      stoichiometric coefficients
    for i in np.arange(n_spec):
        stoich_arr[i+2, 0] = spec_list[i]
        stoich_arr[i+2, 1:] = map(int,spec_stoich[i+1, np.where(is_used)[0]])
    
    # Convert logarithmic (dex) abundances into number densities
    finalstoich_conv = np.empty((n_spec + 2, np.sum(is_used) + 1), \
                                                 dtype=np.object)
    finalstoich_conv[0,0] = 'b'
    finalstoich_conv[0,1:] = map(float, stoich_arr[0,1:])

    # Number densities for elements in the system are equal to 10**(dex)
    finalstoich_conv[0,1:] = 10**(finalstoich_conv[0,1:])
    
    # Elemental abundance is equal to elemental number density divided by
    #           total sum of all elemental number densities in the system
    finalstoich_conv[0,1:] /= sum(finalstoich_conv[0,1:])
    
    # Place converted values back into final header array
    stoich_arr[0] = finalstoich_conv[0]
    
    return stoich_arr


def atm_headarr(spec_list, stoich_data, spec_stoich, atom_arr, q, is_used):
    '''
    This function gathers data needed for TEA to run in a multiple T-P case. 
    These are: elemental abundances, species names, and their stoichiometric
    values. For the list of elements and species used, it takes the abundances
    and stoichiometric values and puts them in the final array. This function
    is run by make_atmheader() and is dependent on results from header_setup().

    Parameters
    ----------
    spec_list: string array
         Array containing names of molecular species.
    stoich_data: array
         Full stoichiometric information from header_setup() for species of 
         interest.
    spec_stoich: float array
         Array containing values from header_setup() for species of interest.
    atom_arr: string array
         Array containing elemental symbols and abundances.
    q: integer
         Current line number in pre-atm file.
    is_used: boolean array
         Array containing booleans to trim stoichiometric data to only the 
         species of interest.

    Returns
    -------
    stoich_arr: array
         Array containing elemental abundances, species names, and their 
         stoichiometric values.
    '''

    # Get number of species and elements used
    n_spec = np.size(spec_list)
    n_atom = np.size(atom_arr[0])
    
    # Allocate final abundance array
    stoich_arr = np.empty((n_spec + 2, np.sum(is_used) + 1), dtype=np.object)
    stoich_arr[0,0] = 'b'
    
    # Get only the abundances used in the species list
    for n in np.arange(np.sum(is_used)):
        # Get list of used elements from species list
        cur_ele = stoich_data[1, np.where(is_used)[0][n] + 1]
        
        # Place used elemental abundances into final abundance array
        for m in np.arange(n_atom):
            if atom_arr[0][m] == cur_ele:
                cur_abn = atom_arr[q][m]
        stoich_arr[0,1+n] = cur_abn

    # Fill in final abundance array
    stoich_arr[1,0] = 'Species'
    stoich_arr[1,1:] = stoich_data[1, np.where(is_used)[0] + 1]
    for i in np.arange(n_spec):
        stoich_arr[i+2, 0] = spec_list[i]
        stoich_arr[i+2, 1:] = map(int,spec_stoich[i+1, np.where(is_used)[0]])
    
    return stoich_arr


def write_header(desc, pressure, temp, stoich_arr, n_spec, g_RT):
    '''
    This function writes a header file that contains all necessary data
    for TEA to run. It is run by make_atmheader() and make_singleheader().

    Parameters
    ----------
    desc: string
         Name of output directory given by user. 
    pressure: float
         Current pressure value.
    temp: float
         Current temperature value.
    stoich_arr: array
         Array containing elemental abundances, species names, and their 
         stoichiometric values.
    n_spec: integer
         Number of species. 
    g_RT: float array
         Array containing chemical potentials for each species at the 
         current T-P.

    Returns
    -------
    None
    '''

    # Comment at top of header file
    header_comment = ("# This is a header file for one T-P. It contains the following data:\n"
    "# pressure (bar), temperature (K), elemental abundances (b, unitless),\n"
    "# species names, stoichiometric values of each element in the species (a),\n"
    "# and chemical potentials.\n\n")

    # Make header directory if it does not exist to store header files
    if not os.path.exists(location_out + desc + '/headers/'): os.makedirs(location_out + desc + '/headers/')

    # Create header file to be used by the main pipeline
    outfile = location_out + desc + '/headers/' + 'header_' + desc + ".txt"

    # Open file to write
    f = open(outfile, 'w+')

    # Write comments and data
    f.write(header_comment)
    f.write(np.str(pressure) + '\n')
    f.write(np.str(temp)     + '\n')

    b_line = stoich_arr[0][0]
    for i in np.arange(np.shape(stoich_arr)[1] - 1):
        b_line += ' ' + np.str(stoich_arr[0][i + 1])
    f.write(b_line + '\n')

    # Retrieve the width of the stoichiometric array
    width = np.shape(stoich_arr)[1]

    # Add title for list of chemical potentials
    g_RT = np.append(["Chemical potential"], g_RT)

    # Add title for species names
    stoich_arr[1][0] = "# Species"

    # Loop over all species and titles
    for i in np.arange(n_spec + 1):
        # Loop over species and number of elements
        for j in np.arange(width):
            # Write species names
            if j == 0:
                f.write(np.str(stoich_arr[i+1][j]).rjust(9) + "  ")
            # Write elemental number density
            else:
                f.write(np.str(stoich_arr[i+1][j]).ljust(3))
        # Write chemical potentials, adjust spacing for minus sign
        if g_RT[i][0] == '-':
            f.write(np.str(g_RT[i]).rjust(13) + '\n')
        else:
            f.write(np.str(g_RT[i]).rjust(14) + '\n')

    f.close()


def make_singleheader(infile, desc, thermo_dir):
    '''
    This is the main function that creates single-run TEA header. It reads
    the input T-P file and retrieves necessary data. It calls the
    header_setup(), single_headarr(), and write_header() functions to create a
    header for the single T-P point. This function is called by the 
    runsingle.py module. 

    Parameters
    ----------
    infile: string
         Name of the input file (single T-P file).         
    desc: string
         Name of output directory given by user.    
    thermo_dir = 'lib/gdata':  string
         Name of directory containing thermodynamic data.

    Returns
    -------
    desc: string
         Name of output directory given by user. 
    pressure: float
         Current pressure value.
    temp: float
         Current temperature value.
    stoich_arr: array
         Array containing elemental abundances, species names, and their 
         stoichiometric values.
    n_spec: integer
         Number of species. 
    g_RT: float array
         Array containing chemical potentials for each species at the 
         current T-P.
    '''

    # Read single input file
    f = open(infile, 'r')

    # Allocate species list
    spec_list = []
    
    # Begin by reading first line of file (l = 0)
    l = 0
    
    # Loop over all lines in input file to retrieve appropriate data
    for line in f.readlines():
        # Retrieve temperature
        if (l == 0):
            temp = np.float([value for value in line.split()][0])
        # Retrieve pressure
        if (l == 1):
            pressure = np.float([value for value in line.split()][0])
        # Retrieve list of species
        if (l > 1):
            val = [value for value in line.split()][0]
            spec_list = np.append(spec_list, val)
        # Update line number
        l += 1
    
    f.close()
    
    # Retrieve number of species used
    n_spec = np.size(spec_list)
    
    # Execute header setup
    stoich_data, spec_stoich, g_RT, is_used =                              \
                        header_setup(temp, pressure, spec_list, thermo_dir)
    
    # Execute single-specific header setup
    stoich_arr = single_headarr(spec_list, stoich_data, spec_stoich, is_used)
    
    # Write header array to file
    write_header(desc, pressure, temp, stoich_arr, n_spec, g_RT)


def make_atmheader(q, spec_list, pressure, temp, atom_arr, desc, \
                   thermo_dir = 'lib/gdata'):
    '''
    This is the main function that creates a header for one T-P of a 
    pre-atm file. It retrieves number of elements and species used for 
    only the q-th T-P point in the pre-atm file. It then calls the 
    header_setup(), atm_headarr(), and write_header() functions to create a 
    header for this point. This function is called by the runatm.py module. 

    Parameters
    ----------
    q: integer
         Current line number in pre-atm file.
    spec_list: string array
         Array containing names of molecular species.
    pressure: float
         Current pressure value.
    temp float
         Current temperature value.
    atom_arr: string array
         Array containing elemental symbols and abundances.
    desc: string
         Name of output directory given by user.    
    thermo_dir = 'lib/gdata':  string
         Name of directory containing thermodynamic data.

    Returns
    -------
    None
    '''

    # Retrieve number of elements and species used
    n_spec   = np.size(spec_list)
    n_atom   = np.size(atom_arr[0])
    
    # Execute header setup
    stoich_data, spec_stoich, g_RT, is_used =                             \
                       header_setup(temp, pressure, spec_list, thermo_dir)
    
    # Execute multiple-specific header setup
    stoich_arr = atm_headarr(spec_list, stoich_data, spec_stoich, atom_arr,\
                                                               q, is_used)
    # Write header array to file
    write_header(desc, pressure, temp, stoich_arr, n_spec, g_RT)

