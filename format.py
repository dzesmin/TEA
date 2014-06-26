
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

import numpy as np
from sys import stdout

# =============================================================================
# This is an auxiliary program that allows each program to read the output of
# the previous step so the data can be used in the next step. It also manages
# the format for each output file and produces both machine-readable and 
# human-readable files. 
#
# It contains the following functions: 
#       readheader(): Reads the header file for the pipeline.
#       readoutput(): Reads output files produced and used by the pipeline.
#           output(): Writes machine readable output. 
#         fancyout(): Writes human-readable output.
# fancyout_results(): Writes final results in human-readable format.
#         printout(): Prints iteration number.
# =============================================================================

def readheader(file):
    '''
    This function reads current header file (one T-P) and returns data common 
    to each step of TEA. It searches only for the required data from the 
    header file and fills out the output arrays appropriately. The function is
    used by balance.py, lagrange.py, lambdacorr.py, and iterate.py.

    Parameters
    ----------
    file: ASCII file
              Header for the current layer in the atmosphere, e.g., one T-P
  
    Returns
    -------
    pressure: float
              Current pressure.
    temp: float
              Current temperature.
    i: integer
              Number of molecular species.
    j: integer
              Number of elements
    speclist: string array
              Array containing names of molecular species.
    a: integer array
              Array of stoichiometric values for each element in the species.
    b: float array
              Array containing elemental abundances - ratio of the number density 
              of a single element to the total sum of elemental number densities 
              in the mixture.
    g_RT: float array
              Array containing chemical potentials for each species at the 
              current T-P.
    '''

    # Open header file to read
    f = open(file, 'r+')

    # Initiate reading the file from zero-th line
    l = 0
    
    # Start indicates when data begins, begin with False so top comment is not
    #       included in data
    start = False

    # Set marker for comments
    comment = '#'

    # Allocates lists of:
    speclist = []     # species names
    a = [[]]          # stoichiometric values
    c = []            # chemical potential

    # Read the file line by line and if correct line is found, assign the data
    #      to corresponding variables and convert to floats/strings
    for line in f.readlines():
        contents = [value for value in line.split()] 
        
        # Boolean to check if the line is blank
        is_blank   = (contents == [])
         
        # Check if non-blank line is comment or data
        if not is_blank: 
            # Boolean to check if the line is comment
            is_comment = (contents[0][0] == comment)
            # If line is not comment or blank start reading
            if not start and contents[0][0].isdigit(): 
                start = l
         
        if start:
            # Skip line if blank or comment
            if is_comment or is_blank: 
                start += 1
            # Read pressure
            elif (l == start):      
                pressure = np.float([value for value in line.split()][0])
            # Read temperature
            elif (l == start+1):     
                temp = np.float([value for value in line.split()][0])
            # Read elemental abundances
            elif (l == start+2):     
                val = [value for value in line.split()]
                b   = [float(u) for u in val[1:]]
            # Read species list, stoichiometry, and chemical potentials
            elif (l == start+3):      
                val = [value for value in line.split()]
                speclist = np.append(speclist, val[0])
                a = [[int(u) for u in val[1:-1]]]
                g_RT = np.float(val[-1])
            elif (l > start+3):      
                val = [value for value in line.split()]
                speclist = np.append(speclist, val[0])
                a = np.append(a, [[int(u) for u in val[1:-1]]], axis=0)
                g_RT = np.append(g_RT, np.float(val[-1]))

        # Go to the next line of the file and check again
        l += 1
    
    # Take the number of species and elements
    i = speclist.size
    j = a.shape[1]
    f.close()
    
    # Convert b array to list
    b = np.array(b).tolist()
    
    return pressure, temp, i, j, speclist, a, b, g_RT


def readoutput(file):
    '''
    This function reads output files made by the balance.py, lagrange.py and
    lambdacorr.py. It reads any iteration's output and returns the data in 
    an array.

    Parameters
    ----------
    file: ASCII file
              Header for the current layer in the atmosphere, i.e. one T-P
  
    Returns
    -------
    header:   string
              Name of the header file used.
    it_num:   integer 
              Iteration number.
    speclist: string array
              Array containing names of molecular species. 
    y: float array
              Array containing initial mole numbers of molecular species for
              current iteration.
    x: float array
              Array containing final mole numbers of molecular species for
              current iteration.
    delta: float array
              Array containing change in mole numbers of molecular species for
              current iteration.
    y_bar: float
              Array containing sum of initial mole numbers for all molecular 
              species for current iteration.
    x_bar: float
              Array containing sum of final mole numbers for all molecular 
              species for current iteration.
    delta_bar: float 
              Change in total mole numbers of all species.
    '''

    # Open output file to read
    f = open(file, 'r')

    # Allocate and fill out data array with all info
    data = []
    for line in f.readlines():
        l = [value for value in line.split()]
        data.append(l)
    
    # Close file 
    f.close()
    
    # Take appropriate data
    header    = data[0][0]                            # header name
    it_num    = np.array(data[1]).astype(np.int)[0]   # iteration number
    speclist  = np.array(data[2]).astype(np.str)      # species list
    y         = np.array(data[3]).astype(np.float)    # initial mole numbers
    x         = np.array(data[4]).astype(np.float)    # final mole numbers
    delta     = np.array(data[5]).astype(np.float)    # difference (x - y)
    y_bar     = np.array(data[6]).astype(np.float)[0] # sum of y_i's
    x_bar     = np.array(data[7]).astype(np.float)[0] # sum of x_i's
    delta_bar = np.array(data[8]).astype(np.float)[0] # difference in sums
    
    return(header, it_num, speclist, y, x, delta, y_bar, x_bar, delta_bar)


def output(datadir, header, it_num, speclist, y, x, delta, y_bar, \
            x_bar, delta_bar, file, doprint = False):
    '''
    This function produces machine-readable output files. The files are saved
    only if saveout = True in TEA_config.py file. The function is used by the
    balance.py, lagrange.py, and lambdacorr.py. The function writes the name of
    the header, current iteration number, species list, starting mole numbers 
    of species for current iteration, final mole numbers of molecular species 
    after the iteration is done, difference between starting and final mole 
    numbers, total sum of initial mole numbers, total sum of final mole numbers
    and the change in total mole numbers of all species. 

    Parameters
    ----------
    datadir:   string
              Current directory where TEA is run.
    header:   string
              Name of the header file used.
    it_num:   integer 
              Iteration number.
    speclist: string array
              Array containing names of molecular species. 
    y: float array
              Array containing initial mole numbers of molecular species for
              current iteration.
    x: float array
              Array containing final mole numbers of molecular species for
              current iteration.
    delta: float array
              Array containing change in mole numbers of molecular species for
              current iteration.
    y_bar: float
              Array containing sum of initial mole numbers for all molecular 
              species for current iteration.
    x_bar: float
              Array containing sum of final mole numbers for all molecular 
              species for current iteration.
    delta_bar: float 
              Change in total mole numbers of all species.
    file: string
              Name of output file to be written.
    doprint:  string
              Parameter in configuration file that allows printing for 
              debugging purposes.

    Returns
    -------
    None
    '''

    # Open file to write
    f = open(file, 'w+')

    # Count number of species
    i = speclist.size
    
    # Write the location of the header file
    f.write(header + '\n')                  # 1st row
    
    # Write current number of iteration
    f.write(np.str(it_num) + '\n')          # 2nd row
    
    # Write species list
    for n in np.arange(i):                 # 3rd row
        f.write(speclist[n] + ' ')
        if n == (i-1): f.write('\n')
    
    # Write starting mole numbers of molecular species for that iteration
    for n in np.arange(i):                 # 4th row
        f.write(np.str(y[n]) + ' ')
        if n == (i-1): f.write('\n')
    
    # Write final mole numbers of molecular species after iteration is done
    for n in np.arange(i):                 # 5th row
        f.write(np.str(x[n]) + ' ')
        if n == (i-1): f.write('\n')
     
    # Write difference between initial and final mole numbers
    for n in np.arange(i):                 # 6th row
        f.write(np.str(delta[n]) + ' ')
        if n == (i-1): f.write('\n')
    
    # Write total sum of initial mole numbers
    f.write(np.str(y_bar) + '\n')           # 7th row
 
    # Write total sum of final mole numbers 
    f.write(np.str(x_bar) + '\n')           # 8th row
 
    # Write difference of total mole numbers
    f.write(np.str(delta_bar) + '\n')       # 9th row
    
    f.close()
   
    # Debugging check
    if doprint == True:
        print('\n\nMade file \'' + file + '\' containing machine data.')


def fancyout(datadir, it_num, speclist, y, x, delta, y_bar, x_bar, delta_bar, \
             file, doprint = False):
    ''' 
    This function produces human readable output files. The files are saved
    only if saveout = True in TEA_config.py file. The function is used by the
    balance.py, lagrange.py, and lambdacorr.py. The function writes the name of
    the header, current iteration number, species list, starting mole numbers 
    of species for current iteration, final mole numbers of molecular species 
    after the iteration is done, difference between starting and final mole 
    numbers, total sum of initial mole numbers, total sum of final mole numbers
    and the change in total mole numbers of all species.  If doprint = True, 
    all data written to the file is presented on-screen.

    Parameters
    ----------
    datadir: string
              Current directory where TEA is run.
    header: string
              Name of the header file used.
    it_num: integer 
              Iteration number.
    speclist: string array
              Array containing names of molecular species. 
    y: float array
              Array containing initial mole numbers of molecular species for
              current iteration.
    x: float array
              Array containing final mole numbers of molecular species for
              current iteration.
    delta: float array
              Array containing change in mole numbers of molecular species for
              current iteration.
    y_bar: float
              Array containing sum of initial mole numbers for all molecular 
              species for current iteration.
    x_bar: float
              Array containing sum of final mole numbers for all molecular 
              species for current iteration.
    delta_bar: float 
              Change in total mole numbers of all species.
    file: string
              Name of output file to be written.
    doprint: string
              Parameter in configuration file that allows printing for 
              debugging purposes.

    Returns
    -------
    None
    '''

    # Open file to write
    f = open(file, 'w+')

    # Write top comment and iteration number
    f.write('This .txt file is for visual use only.  \
             DO NOT USE FOR ITERATIONS!\n')
    f.write('Data for iteration #' + np.str(it_num) + '\n\n')
    
    # Count number of species
    i = speclist.size

    # Loop over all species
    for n in np.arange(i):
        if n == 0:
            # Write labels for columns
            f.write('Species |'.rjust(10) + 'y_i |'.rjust(12) + \
                     'x_i |'.rjust(12) + 'delta \n'.rjust(12)) 

        # Fill out variables
        xs   = '%8.6f'%x[n]         # Final mole numbers
        ys   = '%8.6f'%y[n]         # Initial mole numbers
        ds   = '%8.6f'%delta[n]     # Difference between initial and final
        xbs  = '%8.6f'%x_bar        # Total sum of final mole numbers
        ybs  = '%8.6f'%y_bar        # Total sum of initial mole numbers
        dbs  = '%8.6f'%delta_bar    # Difference of total sums
        name = speclist[n]          # Species name

        # Write mole numbers in aligned columns 
        f.write(name.rjust(8) + ' |' + ys.rjust(10) + ' |' + xs.rjust(10) + \
                ' |' +ds.rjust(10) + '\n')
        
        # Write initial, final, and difference of totals after species data
        if n == (i - 1):
            f.write('\n')
            f.write('y_bar : '.rjust(35) + ybs.rjust(9) + '\n')
            f.write('x_bar : '.rjust(35) + xbs.rjust(9) + '\n')
            f.write('delta_bar : '.rjust(35) + dbs.rjust(9) + '\n')
    f.close()
    
    # Print for debugging purposes
    # Print all the data from the file on the screen 
    if doprint == True:
        f = open(file, 'r+')
        h = 0
        for line in f:
            if h == 0:
                print('Made file \'' + file + '\' containing the following:')
            else:
                line = line.strip('\n')
                print(line)
            h += 1     
        f.close()


def fancyout_results(datadir, header, it_num, speclist, y, x, delta, y_bar, \
                     x_bar, delta_bar, pressure, temp, file, doprint):
    ''' 
    This function produces the final result output for each T-P in the 
    human-readable format. The final mole number for each species is divided 
    by the total mole numbers of all species in the mixture. This gives our 
    final results, which is the mole fraction abundance for each species. 
    This function is called by the iterate.py module.
    
    Parameters
    ----------
    datadir:  string
              Current directory where TEA is run.
    header:   string
              Name of the header file used.
    it_num:   integer 
              Iteration number of last iteration.
    speclist: string array
              Array containing names of molecular species. 
    y: float array
              Array containing initial mole numbers of molecular species for
              first iteration.
    x: float array
              Array containing final mole numbers of molecular species for
              last iteration.
    delta: float array
              Array containing change in mole numbers of molecular species for
              first and last iterations.
    y_bar: float
              Array containing sum of initial mole numbers for all molecular 
              species for first iteration.
    x_bar: float
              Array containing sum of final mole numbers for all molecular 
              species for last iteration.
    delta_bar: float 
              Change in total mole numbers of all species.
    file: string
              Name of output file to be written.
    doprint:  string
              Parameter in configuration file that allows printing for 
              debugging purposes.
    
    Returns
    -------
    None
    '''

    # Open file to read
    f = open(file, 'w+')

    # Write top comment
    f.write('This .txt file is for visual use only.\n')
    f.write('These results are for the "' + header + '" run.\n')
    f.write('Iterations complete after ' + np.str(it_num) + ' runs at ' + \
                          np.str(pressure) + ' atm and ' + np.str(temp) + \
                                           ' K. Final computation:\n\n')
    
    # Count number of species
    i = speclist.size

    # Loop over all species
    for n in np.arange(i):
        if n == 0:
            # Write labels for columns
            f.write('Species |'.rjust(10) + 'Initial x |'.rjust(16) + \
                        'Final x |'.rjust(16) + 'Delta |'.rjust(16) + \
                                         'Final Abun \n'.rjust(16)) 

        # Fill out variables
        xs   = '%8.10f'%x[n]         # Final mole numbers
        ys   = '%8.10f'%y[n]         # Initial mole numbers
        ds   = '%8.10f'%delta[n]     # Difference between initial and final
        xbs  = '%8.10f'%x_bar        # Total sum of final mole numbers
        ybs  = '%8.10f'%y_bar        # Total sum of initial mole numbers
        dbs  = '%8.10f'%delta_bar    # Difference of total sums

        # Divide final result by the total number of moles of the species
        #        in the mixture, making mole fractions
        abn_float = x[n] / x_bar
        abn = '%8.7e'%abn_float

        # Species name
        name = speclist[n]  

        # Write variables in aligned columns
        f.write(name.rjust(8) + ' |' + ys.rjust(14) + ' |' + xs.rjust(14) + \
                           ' |' +ds.rjust(14) + ' |' +abn.rjust(14) + '\n')
        
        # Write initial, final, and difference of totals after species data
        if n == (i - 1):
            f.write('\n')
            f.write('Initial Total Mol : '.rjust(35)   + ybs.rjust(9) + '\n')
            f.write('Final Total Mol : '.rjust(35)     + xbs.rjust(9) + '\n')
            f.write('Change in Total Mol : '.rjust(35) + dbs.rjust(9) + '\n')
    
    f.close()

    # Print for debugging purposes
    # Print all the data from the file on the screen
    if doprint == True:
        f = open(file, 'r+')
        h = 0
        for line in f:
            if h == 0:
                print('Made file \'' + file + '\' containing the following:')
            else:
                line = line.strip('\n')
                print(line)
            h += 1
        
        f.close()


def printout(str, it_num = False):
    '''
    Prints iteration progress number or other information in one line of 
    terminal.

    Parameters
    ----------
    str:    string
            String defining what will be printed.
    it_num: integer 
            Iteration number to be printed. If False, will print out 
            contents of str instead.

    Returns
    -------
    None
    '''

    # Create print-out for terminal that can be overwritten
    stdout.write('\r\n')
    if np.bool(it_num):
        # Print iteration number
        stdout.write(str % it_num)
    else:
        # Print other input
        stdout.write(str)
    
    # Clear printed value to allow overwriting for next
    stdout.flush()

