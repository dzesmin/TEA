# This file (thermal_properties.py) is open-source software under the
# GNU GENERAL PUBLIC LICENSE  Version 2, June 1991

__all__ = [
    'ROOT',
    'Tea_Network',
    'thermo_eval',
]

import os
from pathlib import Path
import warnings

import numpy as np

from . import janaf


ROOT = str(Path(__file__).parents[1]) + os.path.sep


class Tea_Network(object):
    r"""
    TEA chemical network.

    Examples
    --------
    >>> # (First, make sure you added the path to the TEA package)
    >>> import tea.thermal_properties as tea
    >>> import numpy as np

    >>> nlayers = 81
    >>> temperature = np.tile(1200.0, nlayers)
    >>> pressure = np.logspace(-8, 3, nlayers)
    >>> HCNO_molecules = (
    >>>     'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O').split()
    >>> tea_net = tea.Tea_Network(pressure, temperature, HCNO_molecules)

    >>> # Compute heat capacity at current temperature profile:
    >>> cp = tea_net.heat_capacity()
    >>> print(f'Heat capacity (cp/R):\n{cp[0]}')
    Heat capacity (cp/R):
    [5.26408044 9.48143057 4.11030773 6.77638503 7.34238673 4.05594463
     3.72748083 6.3275286  3.79892261 2.49998117 2.49998117 2.50082308
     2.49998117 2.51092596]
    """
    def __init__(self, pressure, temperature, input_species,
        source='janaf'):
        """Tea_Network init."""
        self.pressure = pressure
        self.temperature = temperature
        self.input_species = input_species

        if source == 'janaf':
            network_data = janaf.setup_network(input_species)
        self.species = network_data[0]
        self.elements = network_data[1]
        self._heat_capacity = network_data[2]
        self._gibbs_free_energy = network_data[3]


    def heat_capacity(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._heat_capacity)


    def gibbs_free_energy(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._gibbs_free_energy)


def thermo_eval(temperature, thermo_func):
    """
    Compute the thermochemical property specified by thermo_func at
    at the requested temperature(s).  These can be, e.g., the
    heat_capacity or gibbs_free_energy functions returned by
    setup_network().

    Parameters
    ----------
    temperature: float or 1D float iterable
        Temperature (Kelvin).
    thermo_func: 1D iterable of callable functions
        Functions that return the thermochemical property.

    Returns
    -------
    thermo_prop: 1D or 2D float array
        The provided thermochemical property evaluated at the requested
        temperature(s).
        The shape of the output depends on the shape of the
        temperature input.

    Examples
    --------
    >>> # (First, make sure you added the path to the TEA package)
    >>> import tea.thermal_properties as tea
    >>> import tea.janaf as janaf
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> janaf_data = janaf.setup_network(molecules)
    >>> species = janaf_data[0]
    >>> heat_funcs = janaf_data[2]

    >>> temperature = 1500.0
    >>> temperatures = np.arange(100.0, 4501.0, 10)
    >>> cp1 = tea.thermo_eval(temperature, heat_funcs)
    >>> cp2 = tea.thermo_eval(temperatures, heat_funcs)

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
    >>> plt.figure('Heat capacity', (6.5, 4.5))
    >>> plt.clf()
    >>> plt.subplot(111)
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(temperatures, cp2[:,j], label=label, c=cols[label])
    >>> plt.xlim(np.amin(temperatures), np.amax(temperatures))
    >>> plt.plot(np.tile(temperature,nspecies), cp1, 'ob', ms=4, zorder=-1)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Heat capacity / R')
    >>> plt.legend(loc=(1.01, 0.01), fontsize=8)
    >>> plt.tight_layout()
    """
    temp = np.atleast_1d(temperature)
    ntemp = np.shape(temp)[0]
    nspecies = len(thermo_func)
    thermo_prop= np.zeros((ntemp, nspecies))
    for j in range(nspecies):
        thermo_prop[:,j] = thermo_func[j](temp)
    if np.shape(temperature) == ():
        return thermo_prop[0]
    return thermo_prop
