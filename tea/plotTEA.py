#! /usr/bin/env python

# ******************************* START LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances under thermochemical equilibrium conditions.
# 
# This project was completed with the support of the NASA Earth and
# Space Science Fellowship Program, grant NNX12AL83H, held by Jasmina
# Blecic, Principal Investigator Joseph Harrington. Project developers
# included graduate student Jasmina Blecic and undergraduate M. Oliver
# Bowman.
# 
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# Our intent is to release this software under an open-source,
# reproducible-research license, once the code is mature and the first
# research paper describing the code has been accepted for publication
# in a peer-reviewed journal.  We are committed to development in the
# open, and have posted this code on github.com so that others can test
# it and give us feedback.  However, until its first publication and
# first stable release, we do not permit others to redistribute the code
# in either original or modified form, nor to publish work based in
# whole or in part on the output of this code.  By downloading, running,
# or modifying this code, you agree to these conditions.  We do
# encourage sharing any modifications with us and discussing them
# openly.
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
# 4111 Libra Drive
# Orlando, FL 32816-2385
# USA
# 
# Thank you for testing TEA!
# ******************************* END LICENSE *******************************

from readconf import *

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

# Correct directory names
if location_out[-1] != '/':
    location_out += '/'

def plotTEA():
    '''
    This code plots a figure of temperature vs. abundances for any multiTP 
    final output produced by TEA, RESULT_ATM_FILE. It accepts 2 arguments 
    on the command line: the path to the multiTP TEA output, and the names
    of the species user wants to plot. See notes for the full description
    of arguments. To run the code do:
    ../TEA/tea/plotTEA.py <RESULT_ATM_FILE_PATH> <SPECIES_NAMES>
    See 'Notes' section for an example.

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
    .tea file. Read the species names and column numbers you want to plot.
    Arguments given should be in the following format:
    ../TEA/tea/plotTEA.py <RESULT_ATM_FILE_PATH> <SPECIES_NAMES>

    <RESULT_ATM_FILE> - string
               Path to the atm_file.           
    <SPECIES_NAMES>  - list of strings
               List of species that user wants to plot. Species names should 
               be given with their symbols (without their states) and no
               breaks between species names (e.g., CH4,CO,H2O).

    Example: ../TEA/tea/plotTEA.py ../TEA/doc/examples/multiTP/results/multiTP_Example.tea CO,CH4,H2O,NH3
    The plot is opened once execution is completed and saved in the ./plots/
    subdirectory.

    The lower range on the y axis (mixing fraction) when temperatures are below ~600 K 
    should be set at most -14. 
    '''
    
    # Get plots directory, create if non-existent
    plots_dir = location_out + "/plots/"
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    
    # Counts number of arguments given
    noArguments = len(sys.argv)

    # Prints usage if number of arguments different from 4 or adiabatic profile
    if noArguments != 3:
        print("\nUsage  : ../TEA/tea/plotTEA.py atmfile species(divided by comma, no breaks)")
        print("Example: ../TEA/tea/plotTEA.py ../TEA/doc/examples/multiTP/results/multiTP_Example.tea CO,CH4,H2O,NH3\n")
     
    # Sets the first argument given as the atmospheric file
    filename = sys.argv[1]

    # Sets the second argument given as the species names
    species = sys.argv[2]

    # Open the atmospheric file and read
    f = open(filename, 'r')
    lines = np.asarray(f.readlines())
    f.close()

    # Get molecules names
    imol = np.where(lines == "#SPECIES\n")[0][0] + 1
    molecules = lines[imol].split()
    nmol = len(molecules)
    for m in np.arange(nmol):
        molecules[m] = molecules[m].partition('_')[0]

    # Take user input for species and split species strings into separate strings 
    #      convert the list to tuple
    species = tuple(species.split(','))
    nspec = len(species)

    # Populate column numbers for requested species and 
    #          update list of species if order is not appropriate
    columns = []
    spec    = []
    for i in np.arange(nmol):
        for j in np.arange(nspec):
            if molecules[i] == species[j]:
                columns.append(i+2)
                spec.append(species[j])

    # Convert spec to tuple
    spec = tuple(spec)

    # Concatenate spec with pressure for data and columns
    data    = tuple(np.concatenate((['p'], spec)))
    usecols = tuple(np.concatenate(([0], columns)))

    # Load all data for all interested species
    data = np.loadtxt(filename, dtype=float, comments='#', delimiter=None,    \
                    converters=None, skiprows=8, usecols=usecols, unpack=True)

    # Open a figure
    plt.figure(1)
    plt.clf()
 
    # Set different colours of lines
    colors = 'bgrcmykbgrcmyk'
    color_index = 0

    # Plot all specs with different colours and labels
    for i in np.arange(nspec):
        plt.loglog(data[i+1], data[0], '-', color=colors[color_index], \
                                        linewidth=2, label=str(spec[i]))
        color_index += 1

    # Label the plot
    plt.xlabel('Mixing Fraction', fontsize=14)
    plt.ylabel('Pressure [bar]' , fontsize=14)
    plt.legend(loc='best', prop={'size':10})

    # Temperature range (plt.xlim) and pressure range (plt.ylim)
    plt.ylim(max(data[0]), min(data[0]))     
  
    # Place plot into plots directory with appropriate name 
    plot_out = plots_dir + filename.split("/")[-1][:-4] + '.png'
    plt.savefig(plot_out)  
    plt.close()
    
    # Return name of plot created 
    return plot_out


# Call the function to execute
if __name__ == '__main__':
    # Make plot and retrieve plot's name
    plot_out = plotTEA()
    
    # Open plot
    plot = Image.open(plot_out)
    plot.show()
