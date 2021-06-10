# This file (thermal_properties.py) is open-source software under the
# GNU GENERAL PUBLIC LICENSE
# Version 2, June 1991

import os
from pathlib import Path
import sys
import numpy as np

ROOT = str(Path(__file__).parents[1]) + os.path.sep
sys.path.append(ROOT+'tea')
import thermal_properties as tea


def test_get_janaf_names_single():
    janaf_species = tea.get_janaf_names('H2O')
    assert len(janaf_species) == 1
    assert janaf_species[0] == 'H-064.txt'


def test_get_janaf_names_multiple():
    species = 'H2O e- H+'.split()
    janaf_species = tea.get_janaf_names(species)
    assert len(janaf_species) == len(species)
    assert janaf_species[0] == 'H-064.txt'
    assert janaf_species[1] == 'D-020.txt'
    assert janaf_species[2] == 'H-002.txt'


def test_get_janaf_names_gas_over_ref():
    janaf_species = tea.get_janaf_names('Na')
    assert janaf_species[0] == 'Na-005.txt'


def test_get_janaf_names_ref_when_no_gas():
    janaf_species = tea.get_janaf_names('H2')
    assert janaf_species[0] == 'H-050.txt'


def test_get_janaf_names_missing():
    species = 'H2O H2O+'.split()
    janaf_species = tea.get_janaf_names(species)
    assert janaf_species[0] == 'H-064.txt'
    assert janaf_species[1] is None


def test_read_janaf():
    janaf_file = 'H-064.txt'  # Water
    temps, heat = tea.read_janaf(janaf_file)

    expected_temp = np.array([
         100.  ,  200.  ,  298.15,  300.  ,  400.  ,  500.  ,  600.  ,
         700.  ,  800.  ,  900.  , 1000.  , 1100.  , 1200.  , 1300.  ,
        1400.  , 1500.  , 1600.  , 1700.  , 1800.  , 1900.  , 2000.  ,
        2100.  , 2200.  , 2300.  , 2400.  , 2500.  , 2600.  , 2700.  ,
        2800.  , 2900.  , 3000.  , 3100.  , 3200.  , 3300.  , 3400.  ,
        3500.  , 3600.  , 3700.  , 3800.  , 3900.  , 4000.  , 4100.  ,
        4200.  , 4300.  , 4400.  , 4500.  , 4600.  , 4700.  , 4800.  ,
        4900.  , 5000.  , 5100.  , 5200.  , 5300.  , 5400.  , 5500.  ,
        5600.  , 5700.  , 5800.  , 5900.  , 6000.  ])
    expected_heat = np.array([
        4.00494915, 4.01096277, 4.03994841, 4.04067004, 4.12077143,
        4.23671398, 4.3688933 , 4.50961195, 4.65706586, 4.80933066,
        4.96339955, 5.11590489, 5.26408044, 5.405641  , 5.53902304,
        5.6636252 , 5.77908666, 5.88552769, 5.98342939, 6.07327284,
        6.15553913, 6.23287426, 6.30323358, 6.36806038, 6.42783574,
        6.48316103, 6.53427678, 6.58166409, 6.62568377, 6.66669664,
        6.70494325, 6.74054387, 6.77409985, 6.80537067, 6.83483739,
        6.86250003, 6.88871941, 6.91325497, 6.93670808, 6.95883819,
        6.97976558, 6.99973079, 7.01873382, 7.03677468, 7.05397362,
        7.07045094, 7.08620661, 7.10136093, 7.11579361, 7.12914385,
        7.14297517, 7.15728758, 7.17159999, 7.18579212, 7.19998426,
        7.2141764 , 7.22836854, 7.24256068, 7.25663254, 7.27082468,
        7.28501682,
        ])
    np.testing.assert_allclose(temps, expected_temp)
    np.testing.assert_allclose(heat, expected_heat)


def test_read_janaf_missing_cp_values():
    janaf_file = 'D-020.txt'  # electron
    temps, heat = tea.read_janaf(janaf_file)
    expected_temp = np.array([
         298.15,  300.  ,  350.  ,  400.  ,  450.  ,  500.  ,  600.  ,
         700.  ,  800.  ,  900.  , 1000.  , 1100.  , 1200.  , 1300.  ,
        1400.  , 1500.  , 1600.  , 1700.  , 1800.  , 1900.  , 2000.  ,
        2100.  , 2200.  , 2300.  , 2400.  , 2500.  , 2600.  , 2700.  ,
        2800.  , 2900.  , 3000.  , 3100.  , 3200.  , 3300.  , 3400.  ,
        3500.  , 3600.  , 3700.  , 3800.  , 3900.  , 4000.  , 4100.  ,
        4200.  , 4300.  , 4400.  , 4500.  , 4600.  , 4700.  , 4800.  ,
        4900.  , 5000.  , 5100.  , 5200.  , 5300.  , 5400.  , 5500.  ,
        5600.  , 5700.  , 5800.  , 5900.  , 6000.  ])
    expected_heat = np.array([
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117])
    np.testing.assert_allclose(temps, expected_temp)
    np.testing.assert_allclose(heat, expected_heat)


def test_read_janaf_stoich_from_species_neutral():
    stoich = tea.read_janaf_stoich('H')
    assert len(stoich) == 1
    assert stoich['H'] == 1.0

    stoich = tea.read_janaf_stoich('H2')
    assert len(stoich) == 1
    assert stoich['H'] == 2.0

    stoich = tea.read_janaf_stoich('H2O')
    assert len(stoich) == 2
    assert stoich['H'] == 2.0
    assert stoich['O'] == 1.0


def test_read_janaf_stoich_from_species_ions():
    stoich = tea.read_janaf_stoich('e-')
    assert len(stoich) == 1
    assert stoich['e'] == 1.0

    stoich = tea.read_janaf_stoich('H-')
    assert len(stoich) == 2
    assert stoich['H'] == 1.0
    assert stoich['e'] == 1.0

    stoich = tea.read_janaf_stoich('H+')
    assert len(stoich) == 2
    assert stoich['H'] == 1.0
    assert stoich['e'] == -1.0

    stoich = tea.read_janaf_stoich('H3O+')
    assert len(stoich) == 3
    assert stoich['H'] == 3.0
    assert stoich['O'] == 1.0
    assert stoich['e'] == -1.0


def test_read_janaf_stoich_from_janaf():
    stoich = tea.read_janaf_stoich(janaf_file='H-064.txt')
    assert len(stoich) == 2
    assert stoich['H'] == 2.0
    assert stoich['O'] == 1.0


def test_read_janaf_stoich_from_formula():
    stoich = tea.read_janaf_stoich(formula='H3O1+')
    assert len(stoich) == 3
    assert stoich['H'] == 3.0
    assert stoich['O'] == 1.0
    assert stoich['e'] == -1.0


def test_setup_janaf_network_neutrals():
    molecules = 'H2O CH4 CO CO2 H2 C2H2 C2H4 OH H He'.split()
    
    species, elements, splines, stoich_vals = \
        tea.setup_janaf_network(molecules)

    expected_elements = ['He', 'C', 'H', 'O']
    expected_stoich_vals = np.array([
        [0, 2, 0, 1],
        [1, 4, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 2],
        [0, 2, 0, 0],
        [2, 2, 0, 0],
        [2, 4, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    np.testing.assert_equal(species, molecules)
    np.testing.assert_equal(elements, ['C', 'H', 'He', 'O'])
    np.testing.assert_equal(stoich_vals, expected_stoich_vals)


def test_setup_janaf_network_ions():
    molecules = 'H2O CH4 CO CO2 H2 C2H2 C2H4 OH H He e- H- H+ H2+ He+'.split()
    
    species, elements, splines, stoich_vals = \
        tea.setup_janaf_network(molecules)

    expected_stoich_vals = np.array([
        [ 0,  2,  0,  1,  0],
        [ 1,  4,  0,  0,  0],
        [ 1,  0,  0,  1,  0],
        [ 1,  0,  0,  2,  0],
        [ 0,  2,  0,  0,  0],
        [ 2,  2,  0,  0,  0],
        [ 2,  4,  0,  0,  0],
        [ 0,  1,  0,  1,  0],
        [ 0,  1,  0,  0,  0],
        [ 0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  1],
        [ 0,  1,  0,  0,  1],
        [ 0,  1,  0,  0, -1],
        [ 0,  2,  0,  0, -1],
        [ 0,  0,  1,  0, -1]
    ])

    np.testing.assert_equal(species, molecules)
    np.testing.assert_equal(elements, ['C', 'H', 'He', 'O', 'e'])
    np.testing.assert_equal(stoich_vals, expected_stoich_vals)


def test_setup_janaf_network_missing_species():
    molecules = 'Ti Ti+ TiO TiO2 TiO+'.split()
    species, elements, splines, stoich_vals = \
        tea.setup_janaf_network(molecules)

    expected_stoich_vals = np.array([
        [ 0,  1,  0],
        [ 0,  1, -1],
        [ 1,  1,  0],
        [ 2,  1,  0]
    ])

    np.testing.assert_equal(species, ['Ti', 'Ti+', 'TiO', 'TiO2'])
    np.testing.assert_equal(elements, ['O', 'Ti', 'e'])
    np.testing.assert_equal(stoich_vals, expected_stoich_vals)

