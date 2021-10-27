# This file is open-source software under the GNU GENERAL PUBLIC LICENSE
# Version 2, June 1991

__all__ = [
    'get_filenames',
    'read_file',
    'read_stoich',
    'setup_network',
]

import os
from pathlib import Path

import more_itertools
import numpy as np
import scipy.interpolate as si
import scipy.constants as sc


ROOT = str(Path(__file__).parents[1]) + os.path.sep


def get_filenames(species):
    """
    Convert species names to their respective JANAF file names.

    Parameters
    ----------
    species: String or 1D string iterable
        Species to search.

    Returns
    -------
    janaf_names: 1D string array
        Array of janaf filenames.  If a species is not found,
        return None in its place.

    Examples
    --------
    >>> import tea.janaf as janaf
    >>> species = 'H2O CH4 CO CO2 H2 e- H- H+ H2+ Na'.split()
    >>> janaf_files = janaf.get_filenames(species)
    >>> for mol, jfile in zip(species, janaf_files):
    >>>     print(f'{mol:5}  {jfile}')
    H2O    H-064.txt
    CH4    C-067.txt
    CO     C-093.txt
    CO2    C-095.txt
    H2     H-050.txt
    e-     D-020.txt
    H-     H-003.txt
    H+     H-002.txt
    H2+    H-051.txt
    Na     Na-005.txt
    """
    species = np.atleast_1d(species)

    janaf_dict = {}
    for line in open(f'{ROOT}tea/janaf_conversion.txt', 'r'):
        species_name, janaf_name = line.split()
        janaf_dict[species_name] = janaf_name

    janaf_names = [
        janaf_dict[molec] if molec in janaf_dict else None
        for molec in species
    ]
    return janaf_names


def read_file(janaf_file):
    """
    Read a JANAF file to extract tabulated thermal properties.

    Parameters
    ----------
    janaf_file: 1D string array
        A JANAF filename.

    Returns
    -------
    temps: 1D double array
        Tabulated JANAF temperatures (K).
    heat_capacity: 1D double array
        Tabulated JANAF heat capacity cp/R (unitless).
    gibbs_free_energy: 1D double array
        Tabulated JANAF Gibbs free energy G/RT (unitless).

    Examples
    --------
    >>> import tea.janaf as janaf
    >>> janaf_file = 'H-064.txt'  # Water
    >>> temps, heat, gibbs = janaf.read_file(janaf_file)
    >>> for i in range(5):
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}  {gibbs[i]:.3f}')
    100.00  4.005  -317.133
    200.00  4.011  -168.505
    298.15  4.040  -120.263
    300.00  4.041  -119.662
    400.00  4.121  -95.583

    >>> janaf_file = 'D-020.txt'  # electron
    >>> temps, heat, gibbs = janaf.read_file(janaf_file)
    >>> for i in range(5):  # temperatures with missing values are ignored
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}  {gibbs[i]:.3f}')
    298.15  2.500  -2.523
    300.00  2.500  -2.523
    350.00  2.500  -2.554
    400.00  2.500  -2.621
    450.00  2.500  -2.709
    """
    janaf_data = np.genfromtxt(
        f'{ROOT}/janaf/{janaf_file}',
        skip_header=3, usecols=(0,1,3,5), delimiter='\t',
        filling_values=np.nan,
        unpack=True,
    )
    temps = janaf_data[0]
    heat_capacity = janaf_data[1] / sc.R

    # https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf
    # Page 15
    df_H298 = janaf_data[3][temps==298.15]
    gibbs_free_energy = (-janaf_data[2] + df_H298*1000.0/temps) / sc.R

    idx_valid = np.isfinite(heat_capacity)
    return (
        temps[idx_valid],
        heat_capacity[idx_valid],
        gibbs_free_energy[idx_valid],
    )


def read_stoich(species=None, janaf_file=None, formula=None):
    """
    Get the stoichiometric data from the JANAF data base for the
    requested species.

    Parameters
    ----------
    species: String
        A species name (takes precedence over janaf_file argument).
    janaf_file: String
        A JANAF filename.
    formula: String
        A chemical formula in JANAF format (takes precedence over
        species and janaf_file arguments).

    Returns
    -------
    stoich: Dictionary
        Dictionary containing the stoichiometric values for the
        requested species. The dict's keys are the elements/electron
        names and their values are the respective stoich values.

    Examples
    --------
    >>> import tea.janaf as janaf
    >>> # From species name:
    >>> for species in 'C H2O e- H2+'.split():
    >>>     print(f'{species}:  {janaf.read_stoich(species)}')
    C:  {'C': 1.0}
    H2O:  {'H': 2.0, 'O': 1.0}
    e-:  {'e': 1.0}
    H2+:  {'e': -1, 'H': 2.0}

    >>> # From JANAF filename:
    >>> print(janaf.read_stoich(janaf_file='H-064.txt'))
    {'H': 2.0, 'O': 1.0}

    >>> # Or directly from the chemical formula:
    >>> print(janaf.read_stoich(formula='H3O1+'))
    {'e': -1, 'H': 3.0, 'O': 1.0}
    """
    # Get chemical formula (JANAF format):
    if formula is None and species is not None:
        janaf_file = get_filenames(species)[0]
    if formula is None and janaf_file is not None:
        with open(f'{ROOT}/janaf/{janaf_file}', 'r') as f:
            header = f.readline()
        formula = header.split('\t')[-1]
        formula = formula[0:formula.index('(')]

    if '-' in formula:
        stoich = {'e': 1}
    elif '+' in formula:
        stoich = {'e': -1}
    else:
        stoich = {}

    previous_type = formula[0].isalpha()
    word = ''
    groups = []
    for letter in formula.replace('-','').replace('+',''):
        if letter.isalpha() != previous_type:
            groups.append(word)
            word = ''
        word += letter
        previous_type = letter.isalpha()
    groups.append(word)
    for e, num in more_itertools.chunked(groups,2):
        stoich[e] = float(num)
    return stoich


def setup_network(input_species):
    """
    Extract JANAF thermal data for a requested chemical network.

    Parameters
    ----------
    species: 1D string iterable
        Species to search in the JANAF data base.

    Returns
    -------
    species: 1D string array
        Species found in the JANAF database (might differ from input_species).
    elements: 1D string array
        Elements for this chemical network.
    heat_capacity: 1D list of callable objects
        Functions that return the species's heat capacity: cp/R.
    gibbs_free_energy: 1D list of callable objects
        Functions that return the species's Gibbs free energy, G/RT.
    stoich_vals: 2D integer array
        Array containing the stoichiometric values for the
        requested species sorted according to the species and elements
        arrays.

    Examples
    --------
    >>> import tea.janaf as janaf
    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>>
    >>> species, elements, heat_capacity, stoich_vals = \
    >>>     janaf.setup_network(molecules)
    >>> print(
    >>>     f'species:\n  {species}\n'
    >>>     f'elements:\n  {elements}\n'
    >>>     f'stoichiometric values:\n{stoich_vals}')
    species:
      ['H2O' 'CH4' 'CO' 'CO2' 'NH3' 'N2' 'H2' 'HCN' 'OH' 'H' 'He' 'C' 'N' 'O']
    elements:
      ['C' 'H' 'He' 'N' 'O']
    stoichiometric values:
    [[0 2 0 0 1]
     [1 4 0 0 0]
     [1 0 0 0 1]
     [1 0 0 0 2]
     [0 3 0 1 0]
     [0 0 0 2 0]
     [0 2 0 0 0]
     [1 1 0 1 0]
     [0 1 0 0 1]
     [0 1 0 0 0]
     [0 0 1 0 0]
     [1 0 0 0 0]
     [0 0 0 1 0]
     [0 0 0 0 1]]
    """
    # Find which species exists in data base:
    janaf_species = get_filenames(input_species)
    nspecies = len(input_species)
    idx_missing = np.array([janaf is None for janaf in janaf_species])
    if np.any(idx_missing):
        missing_species = np.array(input_species)[idx_missing]
        print(f'These input species were not found:\n  {missing_species}')

    species = np.array(input_species)[~idx_missing]
    janaf_files = np.array(janaf_species)[~idx_missing]

    nspecies = len(species)
    heat_capacity = []
    gibbs_free_energy = []
    stoich_data = []
    for i in range(nspecies):
        janaf = janaf_files[i]
        janaf_data = read_file(janaf)
        temp = janaf_data[0]
        heat = janaf_data[1]
        gibbs = janaf_data[2]

        heat_capacity.append(si.interp1d(
            temp, heat, fill_value='extrapolate'))
        gibbs_free_energy.append(si.interp1d(
            temp, gibbs, kind='cubic', fill_value='extrapolate'))
        stoich_data.append(read_stoich(janaf_file=janaf))

    elements = []
    for s in stoich_data:
        elements += list(s.keys())
    elements = sorted(set(elements))

    nelements = len(elements)
    stoich_vals = np.zeros((nspecies, nelements), int)
    for i in range(nspecies):
        for key,val in stoich_data[i].items():
            j = elements.index(key)
            stoich_vals[i,j] = val
    elements = np.array(elements)

    return (
        species,
        elements,
        heat_capacity,
        gibbs_free_energy,
        stoich_vals,
    )
