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
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

import readconf as rc


# Read configuration-file parameters:
TEApars, PREATpars = rc.readcfg()
maxiter, savefiles, verb, times, abun_file, location_out, xtol, ncpu = TEApars

# Correct directory names
if location_out[-1] != '/':
    location_out += '/'

def plotTEA():
    '''
    Plots a figure of temperature vs. abundances for any multiTP
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

    Recall that TEA works without adjustments for the temperatures above ~500 K. 
    For temperatures below 500 K, the code produces results with low precision, 
    thus it is not recommended to use TEA below these temperatures.  
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


