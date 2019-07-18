import numpy as np

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


def makeAbun(solar_abun, abun_file, COratio, solar_times=1):
    """
    Makes the abundaces file to be used by TEA.
    The function reads Asplund et al (2009) elemental abundances file
    http://adsabs.harvard.edu/abs/2009ARA%26A..47..481A, (abudances.txt),
    sets the C/O ratio as requested, and/or multiplies the abundances of metal
    elements (all except He and H) by the desired multiplication factor. 
    The C/O ratio is set by fixing the O abundance to the solar value and
    changing the C elemental abundance so C/O ratio is as requested.

    Parameters
    ----------
    solar_abun: String
       Input Solar abundances filename.
    abun_file: String
       Output filename to store the modified elemental abundances.

    Optional parameters
    -------------------
    COratio: Float
       Desired C/O ratio.
    solar_times: Integer
       Multiplication factor for metal elemental abundances (everything
       except H and He).

    Returns
    -------
    None

    Example
    -------
    solar_abun  = 'abundances.txt'
    abun_file   = 'CO1.2abun.txt'
    COratio     = 1.2
    makeAbun(solar_abun, abun_file, COratio)

    Revisions
    ---------
    2015-10-25  Jasmina   Written by.
    """


    # Read the elemental-abundances file:
    f = open(solar_abun, 'r')
    lines = f.readlines()
    f.close()

    # Count the number of elements:
    nelem = len(lines)
    for line in lines:
        if line.startswith("#"):
            nelem -= 1

    # Allocate arrays to put information:
    index  = np.zeros(nelem, int)
    symbol = np.zeros(nelem, '|S2')
    dex    = np.zeros(nelem, np.double)
    name   = np.zeros(nelem, '|S20')
    mass   = np.zeros(nelem, np.double)

    # Store data into the arrays:
    i = 0
    for line in lines:
        if not line.startswith("#"):
            index[i], symbol[i], dex[i], name[i], mass[i] = line.strip().split()
            i += 1

    # Count the number of elements:
    nelem = len(symbol)

    # Scale the metals aundances:
    imetals = np.where((symbol != "H") & (symbol != "He"))
    dex[imetals] += np.log10(solar_times)

    # Calculate C and O abundances based on C/O requested:
    Odex = dex[np.where(symbol == "O")]
    dex[np.where(symbol == "C")] = np.log10(COratio) + Odex
    
    # Save data to file
    f = open(abun_file, "w")
    # Write header
    f.write("# Elemental abundances:\n"
            "# Columns: ordinal, symbol, dex abundances, name, molar mass.\n")
    # Write data
    for i in np.arange(nelem):
      f.write("{:3d}  {:2s}  {:5.2f}  {:10s}  {:12.8f}\n".format(
              index[i], symbol[i], dex[i], name[i], mass[i]))
    f.close()

