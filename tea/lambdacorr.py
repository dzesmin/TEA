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

import format as form


def dF_dlam(s, i, x, y, delta, c, x_bar, y_bar, delta_bar):
    """
    Equation (34) TEA theory document:
      dF(lam)/dlam = sum_i delta_i[(g(T)/RT)_i + ln(P) +
                                   ln(yi+lam*delta_i)/(y_bar+lam*delta_bar)]
    """
    dF_dlam = 0
    for n in np.arange(i):
        dF_dlam += delta[n] * (c[n] + np.log(y[n] + s*delta[n]) -
                               np.log(y_bar + s*delta_bar))
    return dF_dlam


def lambdacorr(it_num, verb, input, info, save_info=None):
    '''
    This module applies lambda correction method (Section 4 in the TEA theory
    document). When input mole numbers are negative, the code corrects them to
    positive values and pass them to the next iteration cycle. The code reads
    the values from the last lagrange output, the information from the header
    file, performs checks, and starts setting basic equations. It defines a
    'smart' range so it can efficiently explore the lambda values from [0,1].
    Half of the range is sampled exponentially, and the other half linearly,
    totalling 150 points. The code retrieves the last lambda value before first
    derivative becomes positive (equation (34) in TEA theory document), and
    corrects negative mole numbers to positive.

    Parameters
    ----------
    it_num: integer
       Iteration number.
    verb: Integer
       Verbosity level (0=mute, 1=quiet, 2=verbose).
    input: List
       A list containing the results/data from the previous
       calculation in lagrange.py or lambdacorr.py:
       the current header directory, current iteration
       number, array of species names, array of initial guess,
       array of non-corrected Lagrange values, and array of
       lambdacorr corrected values.
    info: list
       pressure: atmospheric layer's pressure (bar)
       i: Number of species
       j: Number of elements
       a: Stoichiometric coefficients
       b: Elemental mixing fractions
       g_RT: Species chemical potential
    save_info: List of stuff
       If not None, save info to files.  The list contains:
         [location_out, desc, speclist, temp]

    Returns
    -------
    y: array of floats
       The initial guess of molecular species.
    x_corr:  array of floats
       Final mole numbers of molecular species.
    delta_corr: array of floats
       Array containing change of initial and final mole numbers of
       molecular species for current iteration.
    y_bar: float
       Total initial guess of all molecular species for
       current iteration.
    x_corr_bar: float
       Total sum of the final mole numbers of all molecular species.
    detla_corr_bar: float
       Change in total number of all species.
    verb: Integer
       Verbosity level (0=mute, 1=quiet, 2=verbose).

    Notes
    -----
    The code works without adjustments for the temperatures above ~500 K. 
    For temperatures below 500 K the code produces results with low
    precision, thus it is not recommended to use TEA below 500 K. 
    Setting xtol to 1e-8 and maxinter to 200 is most optimizing. 
    If higher tolerance level is desired (xtol>1e-8), maxium number
    of iterations must be increased. The result can be further improved
    with fine adjustments to the lambda exploration variables 
    'lower' and 'steps' to larger magnitudes 
    (i.e., lower = -100,  steps = 1000). This will lengthen the time 
    of execution.
    '''

    # Suppress nan warnings, as they are used for finding valid minima
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')

    pressure = info[0]
    i        = info[1]
    g_RT     = info[5]

    # Take final values from last iteration, lagrange.py
    y         = input[0]
    x         = input[1]
    delta     = input[2]
    y_bar     = input[3]
    x_bar     = input[4]
    delta_bar = input[5]

    # Create 'c' value, equation (17) TEA theory document
    # c_i = (g/RT)i + ln(P)
    c = g_RT + np.log(pressure)

    # Create the range of lambda values to explore. To speed up finding
    #        the correct lambda value to use, the range is split into two
    #        parts at range_split: the lower exponential range and the
    #        higher linear range. This helps the system to converge faster.
    range_split = 0.5

    # Create exponential range, low_range
    # Exponent parameter that gives a value close to zero for the start of
    #          lambda exploration
    lower = -50

    # Define number of steps to explore exponential range
    steps = 100

    # Create lower exponential range
    low_range = np.exp(np.linspace(lower, 0, steps+1))

    # Create linear, evenly spaced range, high_range
    high_step = 0.01
    high_range = np.arange(0.5, 1 + high_step, high_step)

    # Combine the two ranges to create one overall range for lambda exploration
    smart_range = np.append(low_range[low_range <= range_split], high_range)

    # Set that lambda is not found
    lam_not_found = True

    # Retrieve last lambda value explored before the minimum energy is passed
    for h in smart_range:
        val = dF_dlam(h, i, x, y, delta, c, x_bar, y_bar, delta_bar)
        if val > 0 or np.isnan(val):
            break

        # If lambda found, take lambda and set lambda not found to false
        lam = h
        lam_not_found = False

    # If lambda is not found (break), function F (equation (33) in the TEA
    #    theory document) set the final x mole numbers to the values calculated
    #    in the last iteration output
    if lam_not_found:
        x_corr = y
    else:
        x_corr = y + lam * delta

    # Correct x values given this value of lambda
    x_corr_bar = np.sum(x_corr)
    delta_corr = y - x_corr
    delta_corr_bar = x_corr_bar - y_bar

    if save_info is not None:
        location_out, desc, speclist, temp = save_info
        hfolder = location_out + desc + "/headers/"
        headerfile = "{:s}/header_{:s}_{:.0f}K_{:.2e}bar.txt".format(
                        hfolder, desc, temp, pressure)
        # Create and name outputs and results directories if they do not exist
        datadir   = location_out + desc + '/outputs/'
        datadir = "{:s}/{:s}_{:.0f}K_{:.2e}bar/".format(
                    datadir, desc, temp, pressure)

        # Export all values into machine and human readable output files
        file = '{:s}/lagrange_iteration-{:03d}_machine-read.txt'.format(
                datadir, it_num)
        form.output(headerfile, it_num, speclist, y, x_corr, delta_corr,
                    y_bar, x_corr_bar, delta_corr_bar, file, verb)
        file = '{:s}/lagrange_iteration-{:03d}_visual.txt'.format(
              datadir, it_num)
        form.fancyout(it_num, speclist, y, x_corr, delta_corr, y_bar,
                     x_corr_bar, delta_corr_bar, file, verb)

    return y, x_corr, delta_corr, y_bar, x_corr_bar, delta_corr_bar


