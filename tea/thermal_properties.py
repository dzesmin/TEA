# This file (thermal_properties.py) is open-source software under the
# GNU GENERAL PUBLIC LICENSE
# Version 2, June 1991
#
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

__all__ = [
    # Constants:
    'ROOT',
    # Functions:
    'get_janaf_names',
    'read_janaf',
]

import os
from pathlib import Path

import numpy as np
import scipy.interpolate as si


ROOT = str(Path(__file__).parents[1]) + os.path.sep


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
        Tabulated JANAF heat capacity cp (J K-1 mol-1).

    Examples
    --------
    >>> import thermal_properties as tea
    >>> janaf_file = 'H-064.txt'  # Water
    >>> temps, heat = tea.read_janaf(janaf_file)
    >>> for i in range(5):
    >>>     print(f'{temps[i]:6.2f}  {heat[i]}')
    100.00  33.299
    200.00  33.349
    298.15  33.59
    300.00  33.596
    400.00  34.262

    >>> janaf_file = 'D-020.txt'  # electron
    >>> temps, heat = tea.read_janaf(janaf_file)
    >>> for i in range(5):  # temperatures with missing cp are ignored
    >>>     print(f'{temps[i]:6.2f}  {heat[i]}')
    298.15  20.786
    300.00  20.786
    350.00  20.786
    400.00  20.786
    450.00  20.786
    """
    temps, heat_capacity = np.genfromtxt(
        f'{ROOT}/janaf/{janaf_file}',
        skip_header=3, usecols=(0,1), delimiter='\t',
        filling_values=np.nan,
        unpack=True,
    )
    idx_valid = np.isfinite(heat_capacity)
    return temps[idx_valid], heat_capacity[idx_valid]

