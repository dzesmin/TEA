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


def test_heat_capacity_single_temp():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    species, elements, cp_splines, stoich_vals = \
        tea.setup_janaf_network(molecules)
    temperature = 1500.0
    cp = tea.heat_capacity(temperature, cp_splines)

    expected_cp = np.array([
        5.6636252 , 10.41029396,  4.23563153,  7.02137982,  8.00580904,
        4.19064967,  3.88455652,  6.65454913,  3.95900511,  2.49998117,
        2.49998117,  2.5033488 ,  2.49998117,  2.50707724])
    np.testing.assert_allclose(cp, expected_cp)


def test_heat_capacity_temp_array():
    molecules = 'H2O CH4 CO C He'.split()
    species, elements, cp_splines, stoich_vals = \
        tea.setup_janaf_network(molecules)
    temperatures = np.arange(100.0, 4501.0, 200.0)
    cp = tea.heat_capacity(temperatures, cp_splines)

    expected_cp = np.array([
        [ 4.00494915,  4.00001798,  3.50040662,  2.55831326,  2.49998117],
        [ 4.04067004,  4.29468525,  3.50497697,  2.50623533,  2.49998117],
        [ 4.23671398,  5.57366148,  3.58339455,  2.50214607,  2.49998117],
        [ 4.50961195,  6.95102049,  3.74900958,  2.50106362,  2.49998117],
        [ 4.80933066,  8.13053147,  3.91811251,  2.50070281,  2.49998117],
        [ 5.11590489,  9.0840507 ,  4.05438109,  2.50058253,  2.49998117],
        [ 5.405641  ,  9.83154339,  4.15805586,  2.5011839 ,  2.49998117],
        [ 5.6636252 , 10.41029396,  4.23563153,  2.5033488 ,  2.49998117],
        [ 5.88552769, 10.85854903,  4.2949258 ,  2.5076786 ,  2.49998117],
        [ 6.07327284, 11.20794022,  4.34074957,  2.51513549,  2.49998117],
        [ 6.23287426, 11.48324364,  4.37695154,  2.52559918,  2.49998117],
        [ 6.36806038, 11.70262042,  4.40617773,  2.53894941,  2.49998117],
        [ 6.48316103, 11.87954105,  4.43035247,  2.55470509,  2.49998117],
        [ 6.58166409, 12.02374761,  4.45043795,  2.57226486,  2.49998117],
        [ 6.66669664, 12.14269697,  4.46811799,  2.59090707,  2.49998117],
        [ 6.74054387, 12.24156084,  4.48363312,  2.61003038,  2.49998117],
        [ 6.80537067, 12.32478931,  4.4972239 ,  2.62903341,  2.49998117],
        [ 6.86250003, 12.39526891,  4.50937141,  2.64743508,  2.49998117],
        [ 6.91325497, 12.45540509,  4.52091755,  2.66511512,  2.49998117],
        [ 6.95883819, 12.5071222 ,  4.53102043,  2.68183297,  2.49998117],
        [ 6.99973079, 12.55198379,  4.54100304,  2.69722783,  2.49998117],
        [ 7.03677468, 12.5910723 ,  4.55014374,  2.71141997,  2.49998117],
        [ 7.07045094, 12.62522965,  4.55868307,  2.72440939,  2.49998117]])
    np.testing.assert_allclose(cp, expected_cp)


def test_tea_network_cp():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    HCNO_molecules = (
        'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O').split()
    tea_net = tea.Tea_Network(pressure, temperature, HCNO_molecules)

    cp1 = tea_net.heat_capacity()
    temp2 = np.tile(700.0, nlayers)
    cp2 = tea_net.heat_capacity(temp2)

    expected_cp1 = np.array([
        5.26408044, 9.48143057, 4.11030773, 6.77638503, 7.34238673,
        4.05594463, 3.72748083, 6.3275286 , 3.79892261, 2.49998117,
        2.49998117, 2.50082308, 2.49998117, 2.51092596])
    expected_cp2 = np.array([
        4.50961195, 6.95102049, 3.74900958, 5.96117901, 5.81564946,
        3.69885601, 3.5409384 , 5.4895911 , 3.56763887, 2.49998117,
        2.49998117, 2.50106362, 2.49998117, 2.53053035])

    assert np.shape(cp1) == (nlayers, len(tea_net.species))
    np.testing.assert_allclose(cp1[0], expected_cp1)
    np.testing.assert_allclose(cp2[0], expected_cp2)
    np.testing.assert_equal(tea_net.temperature, temp2)

