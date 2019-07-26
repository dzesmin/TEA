
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


def readatm(atm_file, spec_mark='#SPECIES', tea_mark='#TEADATA'):
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
               Array containing elemental abundances.
    atom_name: string array
               Array containing elemental symbols.
    marker[0]: integer
               Marks line number in pre-atm file where data start.
    '''

    # Get current working directory and pre-atm file name
    file = atm_file

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
    ipres = np.where(data_label == '#Pressure')[0][0]
    itemp = np.where(data_label == 'Temp'     )[0][0]

    # Mark number of columns preceding element columns
    iatom = 2

    # Place data into corresponding arrays
    pres_arr  = data[1:,ipres]
    temp_arr  = data[1:,itemp]
    atom_name = data[0,iatom:]
    atom_arr  = data[1:,iatom:]

    # Number of times TEA will have to be executed for each T-P
    n_runs = data.shape[0]-1

    return n_runs, spec_list, pres_arr, temp_arr, atom_arr, atom_name, marker[0]


