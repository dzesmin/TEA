# This file (thermal_properties.py) is open-source software under the
# GNU GENERAL PUBLIC LICENSE
# Version 2, June 1991

__all__ = [
    # Constants:
    'ROOT',
    # Functions:
    'setup_janaf_network',
    'get_janaf_names',
    'read_janaf',
    'read_janaf_stoich',
    'heat_capacity',
]

import os
from pathlib import Path

import more_itertools
import numpy as np
import scipy.interpolate as si
import scipy.constants as sc


ROOT = str(Path(__file__).parents[1]) + os.path.sep


def setup_janaf_network(input_species):
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
    heat_capacity_splines: 1D list of numpy splines
        Splines sampling the species's heat capacity/R.
    stoich_vals: 2D integer array
        Array containing the stoichiometric values for the
        requested species sorted according to the species and elements
        arrays.

    Examples
    --------
    >>> import thermal_properties as tea
    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>>
    >>> species, elements, splines, stoich_vals = \
    >>>     tea.setup_janaf_network(molecules)
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
    janaf_species = get_janaf_names(input_species)
    nspecies = len(input_species)
    idx_missing = np.array([janaf is None for janaf in janaf_species])
    if np.any(idx_missing):
        missing_species = np.array(input_species)[idx_missing]
        print(f'These input species were not found:\n  {missing_species}')

    species = np.array(input_species)[~idx_missing]
    janaf_files = np.array(janaf_species)[~idx_missing]

    nspecies = len(species)
    heat_capacity_splines = []
    stoich_data = []
    for i in range(nspecies):
        janaf = janaf_files[i]
        temp, heat_capacity = read_janaf(janaf)
        heat_capacity_splines.append(si.interp1d(
            temp, heat_capacity, fill_value='extrapolate'))

        stoich_data.append(read_janaf_stoich(janaf_file=janaf))

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

    return species, elements, heat_capacity_splines, stoich_vals


def get_janaf_names(species):
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
    >>> import thermal_properties as tea
    >>> species = 'H2O CH4 CO CO2 H2 e- H- H+ H2+ Na'.split()
    >>> janaf_species = tea.get_janaf_names(species)
    >>> for mol, janaf in zip(species, janaf_species):
    >>>     print(f'{mol:5}  {janaf}')
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


def read_janaf(janaf_file):
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

    Examples
    --------
    >>> import thermal_properties as tea
    >>> janaf_file = 'H-064.txt'  # Water
    >>> temps, heat = tea.read_janaf(janaf_file)
    >>> for i in range(5):
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}')
    100.00  4.005
    200.00  4.011
    298.15  4.040
    300.00  4.041
    400.00  4.121

    >>> janaf_file = 'D-020.txt'  # electron
    >>> temps, heat = tea.read_janaf(janaf_file)
    >>> for i in range(5):  # temperatures with missing cp are ignored
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}')
    298.15  2.500
    300.00  2.500
    350.00  2.500
    400.00  2.500
    450.00  2.500
    """
    temps, heat_capacity = np.genfromtxt(
        f'{ROOT}/janaf/{janaf_file}',
        skip_header=3, usecols=(0,1), delimiter='\t',
        filling_values=np.nan,
        unpack=True,
    )

    idx_valid = np.isfinite(heat_capacity)
    return temps[idx_valid], heat_capacity[idx_valid]/sc.R


def read_janaf_stoich(species=None, janaf_file=None, formula=None):
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
    >>> import thermal_properties as tea
    >>> # From species name:
    >>> for species in 'C H2O e- H2+'.split():
    >>>     print(f'{species}:  {tea.read_janaf_stoich(species)}')
    C:  {'C': 1.0}
    H2O:  {'H': 2.0, 'O': 1.0}
    e-:  {'e': 1.0}
    H2+:  {'e': -1, 'H': 2.0}

    >>> # From JANAF filename:
    >>> print(tea.read_janaf_stoich(janaf_file='H-064.txt'))
    {'H': 2.0, 'O': 1.0}

    >>> # Or directly from the chemical formula:
    >>> print(tea.read_janaf_stoich(formula='H3O1+'))
    {'e': -1, 'H': 3.0, 'O': 1.0}
    """
    # Get chemical formula (JANAF format):
    if formula is None and species is not None:
        janaf_file = get_janaf_names(species)[0]
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


def heat_capacity(temperature, cp_splines):
    """
    Compute the heat capacity for the input chemical network at the
    requested temperature(s).

    Parameters
    ----------
    temperature: float or 1D float iterable
        Temperature (Kelvin).
    cp_splines: 1D iterable of heat-capacity numpy splines
        Numpy splines containing heat capacity info for species.

    Returns
    -------
    cp: 1D or 2D float ndarray
        The heat capacity (divided by the universal gas constant, R) for
        each species at the requested temperature(s).
        The shape of the output depends on the shape of the temperature input.

    Examples
    --------
    >>> import thermal_properties as tea
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>>
    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> species, elements, splines, stoich_vals = \
    >>>     tea.setup_janaf_network(molecules)

    >>> temperature = 1500.0
    >>> cp1 = tea.heat_capacity(temperature, splines)
    >>> temperatures = np.arange(100.0, 4501.0, 10)
    >>> cp2 = tea.heat_capacity(temperatures, splines)

    >>> cols = {
    >>>     'H': 'blue',
    >>>     'H2': 'deepskyblue',
    >>>     'He': 'olive',
    >>>     'H2O': 'navy',
    >>>     'CH4': 'orange',
    >>>     'CO': 'limegreen',
    >>>     'CO2': 'red',
    >>>     'NH3': 'magenta',
    >>>     'HCN': '0.55',
    >>>     'N2': 'gold',
    >>>     'OH': 'steelblue',
    >>>     'C': 'salmon',
    >>>     'N': 'darkviolet',
    >>>     'O': 'greenyellow',
    >>> }
    >>> nspecies = len(species)
    >>> plt.figure('heat capacity')
    >>> plt.clf()
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(temperatures, cp2[:,j], label=label, c=cols[label])
    >>> plt.xlim(np.amin(temperatures), np.amax(temperatures))
    >>> plt.plot(np.tile(temperature,nspecies), cp1, 'ob', ms=4, zorder=-1)
    >>> plt.legend(loc=(1.01, 0.01), fontsize=8)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Heat capacity / R')
    >>> plt.tight_layout()
    """
    temp = np.atleast_1d(temperature)
    ntemp = np.shape(temp)[0]
    nspecies = len(cp_splines)
    cp = np.zeros((ntemp, nspecies))
    for j in range(nspecies):
        cp[:,j] = cp_splines[j](temp)
    if np.shape(temperature) == ():
        return cp[0]
    return cp

