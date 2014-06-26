
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

def readatm(atm_file, spec_mark='#FINDSPEC', tea_mark='#FINDTEA'):
    '''
    This function reads a pre-atm file and returns data that TEA will use.
    It opens a pre-atmosphere file to find markers for species and TEA data,
    retrieves the species list, reads data below the markers, and fills out
    data into corresponding arrays. It also returns number of runs TEA must
    execute for each T-P. The function is used by runatm.py.

    Parameters
    -----------
    atm_file:  ASCII file
               Pre-atm file that contains species, radius, pressure, 
               temperature, and elemental abundances data.
    spec_mark: string
               Marker used to locate species data in pre-atm file
               (located in the line immediately preceding the data).
    tea_mark:  string
               Marker used to locate radius, pressure, temperature, and 
               elemental abundances data (located in the line immediately
               preceding the data).

    Returns
    -------
    n_runs:    float
               Number of runs TEA will execute for each T-P.
    spec_list: string array
               Array containing names of molecular species.
    radi_arr:  float array
               Array containing radius data.
    pres_arr:  float array
               Array containing pressure data.
    temp_arr:  float array
               Array containing temperature data.
    atom_arr:  string array
               Array containing elemental symbols and abundances.
    marker[0]: integer
               Marks line number in pre-atm file where data start.
    '''

    # Get current working directory and pre-atm file name
    file = os.getcwd() + '/' + atm_file
    
    # Open file to read
    f = open(file, 'r')

    # Read data from all lines in info list
    info = []
    for line in f.readlines():
        l = [value for value in line.split()]
        info.append(l)    
    f.close()

    # Initiate list of species and TEA markers 
    marker = np.zeros(2, dtype=int) 

    # Number of rows in file
    ninfo  = np.size(info)         
    
    # Set marker list to the lines where data start
    for i in np.arange(ninfo):
        if info[i] == [spec_mark]:
            marker[0] = i + 1
        if info[i] == [tea_mark]:
            marker[1] = i + 1
    
    # Retrieve species list using the species marker 
    spec_list  = info[marker[0]]       
    
    # Retrieve labels for data array
    data_label = np.array(info[marker[1]]) 

    # Number of labels in data array
    ncols      = np.size(data_label) 
 
    # Number of lines to read for data table (inc. label)  
    nrows      = ninfo - marker[1]     
    
    # Allocate data array
    data = np.empty((nrows, ncols), dtype=np.object)
    
    # Fill in data array
    for i in np.arange(nrows):
        data[i] = np.array(info[marker[1] + i])
    
    # Take column numbers of non-element data
    iradi = np.where(data_label == 'Radius'  )[0][0]
    ipres = np.where(data_label == 'Pressure')[0][0]
    itemp = np.where(data_label == 'Temp'    )[0][0]

    # Mark number of columns preceding element columns
    iatom = 3 
    
    # Place data into corresponding arrays
    radi_arr = data[:,iradi]      
    pres_arr = data[:,ipres]      
    temp_arr = data[:,itemp]      
    atom_arr = data[:,iatom:]    
    
    # Number of times TEA will have to be executed for each T-P
    n_runs = data.shape[0]  
    
    return n_runs, spec_list, radi_arr, pres_arr, temp_arr, atom_arr, marker[0]


