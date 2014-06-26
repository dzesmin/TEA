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
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

def plot_TEA():
    '''
    This code plots a figure of temperature vs. abundances for any atm_file
    produced by TEA. It accepts 3 arguments on the command line: the atm_file
    name, the names of the species user wants to plot, and the column numbers
    from the atm_file where the data of interested species are stored. See
    notes for the full description of arguments. To run the code do:
    plot_TEA.py <RESULT_ATM_FILE> <SPECIES_NAMES> <COLUMNS>
    Example: plot_TEA.py results/atm_Example/atm_Example.dat CO,CH4,H2O,N2,NH3 8,9,10,11,12 

    Parameters
    ----------
    None

    Returns
    -------
    plot_out - string
               Name and location of output plot.

    Notes
    -----
    To plot the results, go to the results/ directory and open final
    .dat file. Read the species names and column numbers you want to plot.
    Arguments given should be in the following format:

    plot_TEA - plot_TEA.py
    filename - string
               Full name of the atm_file with the location.           
    species  - list of strings
               List of species that user wants to plot. Species names should 
               be given with their symbols (without their states) and no
               breaks between species names (e.g., CH4,CO,H2O).
    columns  - list of integers
               List of column numbers of species data that user wants to plot.
               First column in the result atm_file starts with 0 
               (e.g., temperature array is stored in column 2). 
               Number should be written without breaks (e.g., 8,9,10). 

    Example: plot_TEA.py results/atm_Example/atm_Example.dat CO,CH4,H2O,N2,NH3 8,9,10,11,12
    
    The plot is opened once execution is completed and saved in the plots/
    subdirectory.

    The lower range on the y axis (mixing fraction) should be set at most -14 
    (the maximum precision of the TEA code). The lower range on the x axis
    (temperature) is currently set to 200 K, although the best precision is
    reached above 1000 K.
    '''
    
    # Get plots directory, create if non-existent
    plots_dir = "plots/"
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    
    # Counts number of arguments given
    noArguments = len(sys.argv)

    # Prints usage if number of arguments different from 4 or adiabatic profile
    if noArguments != 4:
        print '\nUsage: plot_TEA.py atmfile species(divided by comma, no breaks) \
                        columns(devided by comma, no breaks)'
        print '\nExample:\nplot_TEA.py Example_atm_file.dat CO,CH4,H2O 8,9,10 \n'
        return
     
    # Sets the first argument given as the atmospheric file
    filename = sys.argv[1]

    # Sets the second argument given as the species names
    species = sys.argv[2]

    # Sets the third argument given as the column numbers
    colums = sys.argv[3]

    # Take user input for species and split species strings into separate strings 
    #      convert the list to tuple
    species = tuple(species.split(','))

    # Take user input for column numbers and split into separate numbers
    #      convert to integer and tuple
    colums = colums.split(',')
    columns = tuple(map(int, colums))

    # Concatenate species with temperature for data and columns
    data    = tuple(np.concatenate((['T'], species)))
    usecols = tuple(np.concatenate(([2], columns)))

    # Load all data for all interested species
    data = np.loadtxt(filename, dtype=float, comments='#', delimiter=None,    \
                    converters=None, skiprows=13, usecols=usecols, unpack=True)

    # Count number of species
    spec = len(species)

    # Include H2O condensate below 273 K
    for i in np.arange(spec):
        if species[i]=='H2O':
            data[i+1][data[0] < 273] = 1e-100
        elif species[i]=='TiO':
            data[i+1][data[0] < 1930] = 1e-100
        elif species[i]=='VO':
            data[i+1][data[0] < 1710] = 1e-100

    # Open a figure
    plt.figure(1)
    plt.clf()
 
    # Set different colours of lines
    colors = 'bgrcmykbgrcmyk'
    color_index = 0

    # Plot all species with different colours and labels
    for i in np.arange(spec):
        plt.plot(data[0], np.log10(data[i+1]), '-', color=colors[color_index], \
                                                       label=str(species[i]))
        color_index += 1

    # Label the plot
    plt.title(filename.split("/")[-1][:-4], fontsize=14)
    plt.xlabel('T [K]'                    , fontsize=14)
    plt.ylabel('log10 Mixing Fraction'    , fontsize=14)
    plt.legend(loc='best', prop={'size':10})

    # Temperature range (plt.xlim) and pressure range (plt.ylim)
    plt.ylim(-10, -2)     
    plt.xlim(200, 3000)
    
    # Place plot into plots directory with appropriate name 
    plot_out = plots_dir + filename.split("/")[-1][:-4] + '.png'
    plt.savefig(plot_out)  
    plt.close()
    
    # Return name of plot created 
    return plot_out


# Call the function to execute
if __name__ == '__main__':
    # Make plot and retrieve plot's name
    plot_out = plot_TEA()
    
    # Open plot
    plot = Image.open(plot_out)
    plot.show()
