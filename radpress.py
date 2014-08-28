
# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
# 
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Lead scientist and coder Jasmina Blecic, 
# assistant coder for the first pre-release Oliver M. Bowman.   
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

import reader as rd
from makeheader import *

# ===========================================================================
# This module calculates radii for each pressure of the final atm-file. It
# consists of four functions: 
# get_g()            calculates surface gravity, by taking the necessary 
#                    info from the tepfile provided
# radpress()         calculates radii for each pressure in the 
#                    atmosphere, by taking a tepfile, temperate, pressure and 
#                    mean molecular mass array
# mean_molar_mass()  calculates mean molecular mass array for all lines
#                    of the final atm file
# rad()              calls mean_molar_mass() and radpress() function and
#                    returns the final radii array
#
# The module is called by runatm.py. 
# ===========================================================================
 
# 2014-07-04 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version


# reads the tep file and calculates surface gravity
def get_g(tepfile):
    '''
    Calculates planetary surface gravity. Calls tep reader and 
    gets data needed for calculation (g = G*M/r^2). Returns
    surface gravity in m/s^2 and surface radius in km.

    Parameters
    ----------
    tepfile: tep file, ASCII file
 
    Returns
    -------
    Teff: float 

    Example
    -------
    tepfile = "WASP-43b.tep"
    planet_Teff(tepfile)

    Revisions
    ---------
    2014-06-11 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # universal gravitational constant
    G = 6.67384e-11 # m3/kg/s2 or Nm2/kg2

     # opens tepfile to read and get data
    tep = rd.File(tepfile)

    # get planet mass in Mjup
    planet_mass = tep.getvalue('Mp')
    planet_mass = np.float(planet_mass[0])

    # get planet radius in units of Rjup
    planet_rad = tep.getvalue('Rp')
    planet_rad = np.float(planet_rad[0])

    # conversion to kg and km
    Mjup = 1.89813e27    # kg
    Rjup = 71492.        # km

    # mass and radius of the star in kg and m for g calculation
    Mp = planet_mass * Mjup  # kg
    Rp = planet_rad  * Rjup * 1000. # m

    # planet surface gravity
    # g = G*Mp/ Rp^2 in m/s^2
    g = G * Mp / (Rp**2)

    # convert radius back to km
    Rp = Rp / 1000. 

    return g, Rp


# calculates radii for each pressure in the atmosphere
def radpress(tepfile, temp, mu, pres):
    '''
    Given a pressure in bar, temperature in K, mean molecular
    mass in g/mol, the function calculates radii in km for each 
    layer in the atmosphere. It declares constants, calls get_g()
    function, allocates array for radii, and calculates radii for
    each pressure. The final radii array is given in km.

    Parameters
    ----------
    g: float
       Surface gravity in m/s^2.
    R0: float
       Surface radius in km.
    T: array of floats
       Array containing temperatures in K for each layer in the atmosphere.
    mu: array of floats
       Array containing mean molecular mass in g/mol for each layer 
       from in the atmosphere.
    P: array of floats
       Array containing pressures in bar for each layer in the atmosphere.

    Returns
    -------
    rad: array of floats
        Array containing radii in km for each pressure layer in the atmosphere.

    Revisions
    ---------
    2014-07-04 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # Define physical constants
    Boltzmann = 1.38065e-23 # J/K == (N*m) / K = kg m/s^2 * m / K
    Avogadro  = 6.022e23    # 1/mol

    # Calculate surface gravity
    g, Rp = get_g(tepfile)   # g in m/s^2 and Rp in km
 
    # Number of layers in the atmosphere
    n = len(pres)

    # Allocate array of radii
    rad = np.zeros(n)
     
    # Reverse the order of pressure and temperature array, so it starts from 
    #         the bottom of the atmosphere where first radii is defined as Rp
    pres = pres[::-1]
    temp = temp[::-1]

    # Andrews:"Introduction to Atmospheric Physics" page 26
    # rad2 - rad1 = (R * T1)/g * ln(p1/p2)
    # R = R*/mu, R* - universal gas constant, mu - mean molecular mass
    # R* = Avogadro * Boltzmann
    # rad2 = rad1 + (Avogadro * Bolztmann * T1) / (g * mu1) * ln(p1/p2)
    # in the second term no conversion needed 
    for i in np.arange(n):
        if i == 0:
            rad[0] = Rp
        else:
            rad[i] = rad[i-1] + (Avogadro * Boltzmann * temp[i-1]  * np.log(pres[i-1]/pres[i])) / (g * mu[i-1])

    # Reverse the order of calculated radii to write them in the right order
    #         in pre-atm file
    rad = rad[::-1]

    return rad


# calculates mean molecular mass for each line (T-P) of the final atm file
def mean_molar_mass(atom_arr, temp, pres, spec_list, thermo_dir, n_runs, q, abun_matrix):
    '''
    This function calculates mean molecular mass for each line (T-P) 
    of the final atm file. It gets the current working directory name, 
    and the basic abundances file (default: abundances.txt) to read
    the elemental weights for the elemental species of interest. Then, 
    from the current header file it takes the stoichiometric values 
    for each species, and calculates mean molecular mass, by multiplying
    the total species weight with the abundances value found for that
    T-P point. It gives an array of mean molecular masses for all lines
    of the final atm file (all T-P's).

    Parameters
    ----------
    atom_arr: array of strings
              Array containing names of the input elemental species.
    temp: float
              Current temperature.
    pres: float
              Current pressure.
    spec_list: array of strings
              Array containing names of the output species.
    thermo_dir: string
              Directory name containing thermochemical data.
    n_runs: integer
              Number of runs TEA will be executed.
    q: integer
              Current number of run.
    abun_matrix: array of floats
              2D matrix containing abundances values for all species and all T-P's.

    Returns
    -------
    mu: array of floats
              Array containing mean molecular masses for all T-P's.

    Revisions
    ---------
    2014-07-14 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # Set up locations of the basic abundances file
    cwd = os.getcwd() + '/'
    abun_file = cwd + "abundances.txt"

    # Read abundance data and convert to an array
    f = open(abun_file, 'r')
    abundata = []
    for line in f.readlines():
        if line.startswith('#'):
            continue
        else:
            l = [value for value in line.split()]
            abundata.append(l)
    abundata = np.asarray(abundata)
    f.close()

    # Take number of elements
    n_elem = len(atom_arr[0])

    # Trim abundance data only for elements in the current run
    data_slice = np.zeros(abundata.shape[0], dtype=bool)
    for i in np.arange(n_elem):
        full_el_name = spec_list[i]
        element_name = full_el_name.split("_")[0][:]
        data_slice += (abundata[:,1] == element_name)

    # List of elements of interest and their corresponding data
    abun_trim = abundata[data_slice]

    # Take only weights from abundance data and convert them to floats
    weights = map(float, abun_trim[:,4])

    # Execute header setup from makeheader.py
    stoich_data, spec_stoich, g_RT, is_used =                             \
                   header_setup(temp, pres, spec_list, thermo_dir)
  
    # Execute atm header from makeheader.py and take stoihiometric data
    stoich_arr = atm_headarr(spec_list, stoich_data, spec_stoich, atom_arr,\
                                                               q, is_used)

    # Take the list of stoihiometric values for each species from the header 
    stoich_val = stoich_arr[2:,1:]

    # Multile elemental molar mass with stoihiomtric value
    elem_weight = weights * stoich_val

    # Allocate space for the sum of elemental weights
    spec_weight = np.zeros(np.size(spec_list))
    # Sum all elemental weights to get the molar mass of the species
    for i in np.arange(np.size(spec_list)):
            spec_weight[i] = sum(elem_weight[i])

    # Multiply species weight with the fractional abundance
    mu_array = spec_weight * abun_matrix[1:]

    # Allocate space for mu, final mean molar mass for each T-P
    mu = np.zeros(n_runs-1)
    # Sum of all species weight fraction for one T-P
    for i in np.arange(n_runs - 1):
        mu[i] = sum(mu_array[i])

    return mu


# wrapper for radpress() and mean_molar_mass() functions, gives final radii
def rad(tepfile, atom_arr, temp, pres, spec_list, thermo_dir, n_runs, q, abun_matrix, temp_arr, pres_arr):
    '''
    This function calculates radii for each pressure of the atm file. It takes 
    the mean molecular mass array calculated by mean_molar_mass() function, 
    the temperature and pressure array from the pre-atm file, it locates
    the tepfile of interest and call radpress() function to calculate radii.

    Parameters
    ----------
    atom_arr: array of strings
              Array containing names of the input elemental species.
    temp: float
              Current temperature.
    pres: float
              Current pressure.
    spec_list: array fo strings
              Array containing names of the output species.
    thermo_dir: string
              Directory name containing thermochemical data.
    n_runs: integer
              Number of runs TEA will be executed.
    q: integer
              Current number of run.
    abun_matrix: array of floats
              2D matrix containing abundances values for all species and all T-P's.
    temp_arr: array of floats
              Array containing all temperatures from the pre-atm file plus string "T"
              that distinguishes the temperature columns.
    pres_arr: array of floats
              Array containing all pressure from the pre-atm file plus string "P"
              that distinguishes the temperature columns.

    Returns
    -------
    mu: array of floats
              Array containing mean molecular masses for all T-P's.

    Revisions
    ---------
    2014-07-04 0.1  Jasmina Blecic, jasmina@physics.ucf.edu   Original version
    '''

    # Call mean_mola_mass() function to calculate mu for each T-P
    mu = mean_molar_mass(atom_arr, temp, pres, spec_list, thermo_dir, n_runs, q, abun_matrix)

    # Take temperature array from pre-atm file
    temp1 = map(float, temp_arr[1:])

    # Take pressure array from pre-atm file
    pres1 = map(float, pres_arr[1:])

    # Locate tepfile of interest
    #tepfile = 'inputs/tepfile/HD209458b.tep'

    # Call radpress() to calculate radii
    rad = radpress(tepfile, temp1, mu, pres1)

    return rad


