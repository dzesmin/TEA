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
import string
import re

# ============================================================================
# This program sets and/or executes the pre-pipeline TEA routines readJANAF.py 
# and makestoich.py. It consists of two functions: comp() and setup(). The 
# comp() function counts the number of each element in a chemical species, 
# while the setup() function reads JANAF tables and allows sharing of common 
# routines for readJANAF.py and makestoich.py. If executed as "prepipe.py", it
# will run both routines. If desired, user can run each routine separately.
# ============================================================================

def comp(specie):
    '''
    Species counting function. Counts the number of each element in a chemical
    species. Takes in a string of a chemical species (i.e., "H2O") and returns
    an array containing every element with corresponding counts found in that
    species. Called by makestoich.py and the setup() function. If desired, user
    can return stoichiometric array containing only the elements found in the 
    input species. Otherwise, it returns the full array of all 113 available 
    elemental stoichiometric data.

    Parameters
    ----------
    specie : string
             Chemical species in the format "atomic symbol, count, etc" such
             that the number of counts is always directly following the
             corresponding atomic symbol. The 'species' string can contain
             redundancies, but not parentheses. Names should match the 
             "JCODE" formulas listed in the NIST JANAF Tables listed at: 
             kinetics.nist.gov/janaf/formula.html.
             If 'species' is "NOUSE", function will return an array of all
             0 counts.  
 
    Returns
    -------
    elements : 2D array
               Array containing three columns of equal length: the first 
               column is a full list of all elements' atomic numbers from
               deuterium (#0) up to copernicium (#112), the second column
               contains the corresponding atomic symbol, and the third column
               counts of each of these elements found in the input species.

    Notes
    ------
    "Weight" in the code is the count of each element occurrence in the 
    species, and the sum of all weights for that element is the stoichiometric 
    coefficient (i.e., ClSSCl that appears in JANAF tables has weight 1 for
    first occurrence of Cl, weight 1 for first occurrence of S, and the final
    stoichiometric values of Cl is 2, and for S is 2).

    Capitals imply the beginning of an atomic symbol (i.e., He, Be,  Mg, etc)
    and the digit indicates the count of the element (weight) preceding it in
    the species (i.e, H2 has 2 H's and Li4 has 4 Li's).
    '''

    # List of all elements' symbols from periodic table  
    # Start with Deuterium, end with Copernicium.
    symbols = np.array([
    'D',
    'H',
    'He',
    'Li',
    'Be',
    'B',
    'C',
    'N',
    'O',
    'F',
    'Ne',
    'Na',
    'Mg',
    'Al',
    'Si',
    'P',
    'S',
    'Cl',
    'Ar',
    'K',
    'Ca',
    'Sc',
    'Ti',
    'V',
    'Cr',
    'Mn',
    'Fe',
    'Co',
    'Ni',
    'Cu',
    'Zn',
    'Ga',
    'Ge',
    'As',
    'Se',
    'Br',
    'Kr',
    'Rb',
    'Sr',
    'Y',
    'Zr',
    'Nb',
    'Mo',
    'Tc',
    'Ru',
    'Rh',
    'Pd',
    'Ag',
    'Cd',
    'In',
    'Sn',
    'Sb',
    'Te',
    'I',
    'Xe',
    'Cs',
    'Ba',
    'La',
    'Ce',
    'Pr',
    'Nd',
    'Pm',
    'Sm',
    'Eu',
    'Gd',
    'Tb',
    'Dy',
    'Ho',
    'Er',
    'Tm',
    'Yb',
    'Lu',
    'Hf',
    'Ta',
    'W',
    'Re',
    'Os',
    'Ir',
    'Pt',
    'Au',
    'Hg',
    'Tl',
    'Pb',
    'Bi',
    'Po',
    'At',
    'Rn',
    'Fr',
    'Ra',
    'Ac',
    'Th',
    'Pa',
    'U',
    'Np',
    'Pu',
    'Am',
    'Cm',
    'Bk',
    'Cf',
    'Es',
    'Fm',
    'Md',
    'No',
    'Lr',
    'Rf',
    'Db',
    'Sg',
    'Bh',
    'Hs',
    'Mt',
    'Ds',
    'Rg',
    'Cn' ])
    
    # Count elements
    n_ele = np.size(symbols)
    
    # Create 2D array containing all symbols, atomic numbers, and counts
    elements = np.empty((n_ele, 3), dtype=np.object)
    elements[:, 0] = np.arange(n_ele)
    elements[:, 1] = symbols
    elements[:, 2] = 0
    
    # Scenario for returning empty array
    if specie == 'NOUSE':
        return elements
    
    # Allocate string length and array of booleans to indicate if characters
    #          are capitals or digits
    chars   = len(specie)
    iscaps  = np.empty(chars, dtype=np.bool)
    isdigit = np.empty(chars, dtype=np.bool)
    
    # Check each character in string to fill in boolean arrays for capitals
    #       or digits; 
    for i in np.arange(len(specie)):
        iscaps[i] = (re.findall('[A-Z]', specie[i]) != [])
        isdigit[i] = specie[i].isdigit()
    
    # Indicator for ending each count and blank results array
    endele = True
    result = [[]]
    
    # Loop over all characters in species string
    for i in np.arange(len(specie)):  
        # Start tracking new element
        if endele == True:
            ele = ''
            weight = 0
            endele = False
        
        # Check if character is a letter, if so add to element name
        if (isdigit[i] == False): 
            ele += specie[i]
        
        # Check if character is a digit, if so, make this the element's weight
        if isdigit[i] == True:
            weight = np.int(specie[i])
        
        # Check if element name ends (next capital is reached) 
        #       and if no weight (count) is found, set it to 1
        if (isdigit[i] == False and                \
           (iscaps[i+1:i+2] == True or i == chars-1)):
            weight = 1
        
        # If next element is found or if end of species name is reached 
        #    (end of string), stop tracking
        if (iscaps[i+1:i+2] == True or i == chars-1): 
            endele = True
        
        # End of element has been reached, so output weights of element 
        #     into elemental array
        if endele == True:
            # Locate element in element array and add the weight found
            index = np.where(elements[:, 1] == ele)[0]
            elements[index, 2] += weight

            # Create array containing only the elements used in this run 
            # This is not explicitly returned but can easy be if desired
            if result == [[]]:
                result = np.append(result, [[ele, np.float(weight)]], axis=1)
            else:
                result = np.append(result, [[ele, np.float(weight)]], axis=0)
    
    # Return full array of elements and stoichiometry
    return elements


def setup(raw_dir = 'rawtables', thermo_dir = 'gdata', \
          stoich_dir = 'stoichcoeff/', stoich_out = 'stoich.txt', \
          abun_file = 'abundances.txt'):
    '''
    This routine reads raw JANAF tables placed in the appropriate directory 
    (default: rawtables/) and extracts partial thermodynamic and stoichiometric
    data. It serves as a setup for the readJANAF.py and makestoich.py routines.
    The program is given the name of the raw JANAF tables directory,
    thermodynamic output directory, and stoichiometric output directory, as
    well as the names of the output stoichiometric file and the pre-written 
    file containing abundance data (default: abundances.txt). 
    It takes number of elements from comp() and loops over all JANAF data and
    abundance data, performing various thermodynamic calculations and
    stoichiometric functions on the appropriate species of interest.

    Parameters
    ----------
    raw_dir = 'rawtables': string
           Directory name of raw JANAF tables.
    thermo_dir = 'gdata':  string
           Direcoty name of thermodynamic data output.
    stoich_dir = 'stoichcoeff/': string
           Directory name of stoichiometric output.
    stoich_out = 'stoich.txt': string
           Name of output file to contain stoichiometric data.
    abun_file = 'abundances.txt': string
           Name of file that contains elemental abundance data.

    Returns
    -------
    raw_dir: string
           Directory name of raw JANAF tables.
    thermo_dir: string
           Directory name of thermodynamic data output. 
    stoich_dir: string
           Directory name of stoichiometric output. 
    stoich_out: string
           Name of output file to contain stoichiometric data.
    abun_file: string
           Name of file that contains elemental abundance data.           
    n_ele: integer
           Number of all elements' symbols from comp()'s element list. 
    n_JANAF: integer
           Number of JANAF files in thermo_dir directory.
    species: list
           List containing all species data from JANAF tables.
    JANAF_files: list
           List containing all raw JANAF tables placed in raw_dir directory.
    '''
    # Correct directory names
    if raw_dir[-1] != '/':
        raw_dir += '/'

    if thermo_dir[-1] != '/':
        thermo_dir += '/'

    if stoich_dir[-1] != '/':
        stoich_dir += '/'

    # Get number of elements used
    n_ele = np.shape(comp(''))[0]

    # Get all JANAF file names and make empty array for species data extraction
    JANAF_files = os.listdir(raw_dir)
    n_JANAF = np.size(JANAF_files)
    species = np.empty((n_JANAF, 4), dtype='|S50')

    # Loop over all JANAF files to get specie names
    for i in np.arange(n_JANAF):
        infile = raw_dir + JANAF_files[i]
        f = open(infile, 'r')
        line = [value for value in f.readline().split()]
        f.close()
        string = ' '.join(line)

        # Get the specie name from the JANAF file
        specie = re.search(' \((.*?)\) ', string).group(1)

        # Correct name for ions
        start = string.find(' (' + specie)
        if specie[-1] == '+':
            specie = specie.strip('+') + '_ion_p'
        if specie[-1] == '-':
            specie = specie.strip('-') + '_ion_n'
        
        # Retrieve chemical formula
        species[i, 0] = specie
        
        # Add additional string for the type of the chemical compound
        compound_name = string[: start].split()
        add_string = compound_name[-1]
        species[i, 3] = add_string     
        
    # Loop over JANAF files to get species stoichiometric coefficients
    #      and states
    for i in np.arange(n_JANAF):
        infile = raw_dir + JANAF_files[i]
        f = open(infile, 'r')
        line = [value for value in f.readline().split()]
        f.close()

        # Retrieve stoichiometry and state
        string = ' '.join(line)
        state = re.search('\((.*?)\)', line[-1]).group(0)
        stoch = line[-1].strip(state)
        outstate = re.search('\((.*?)\)', state).group(1).replace(',', '-')
        stoch = stoch.strip('+')
        stoch = stoch.strip('-')
        species[i, 1] = stoch               # stoichiometry
        species[i, 2] = outstate            # state
    
    return raw_dir, thermo_dir, stoich_dir, stoich_out, abun_file, \
           n_ele, n_JANAF, species, JANAF_files


# Execute full pre-pipeline  
# Run readJANAF.py
if __name__ == '__main__':
    from readJANAF import *

# Run makestoich.py
if __name__ == '__main__':
    from makestoich import *
