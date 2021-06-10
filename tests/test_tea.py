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
        33.299, 33.349, 33.59 , 33.596, 34.262, 35.226, 36.325, 37.495,
        38.721, 39.987, 41.268, 42.536, 43.768, 44.945, 46.054, 47.09 ,
        48.05 , 48.935, 49.749, 50.496, 51.18 , 51.823, 52.408, 52.947,
        53.444, 53.904, 54.329, 54.723, 55.089, 55.43 , 55.748, 56.044,
        56.323, 56.583, 56.828, 57.058, 57.276, 57.48 , 57.675, 57.859,
        58.033, 58.199, 58.357, 58.507, 58.65 , 58.787, 58.918, 59.044,
        59.164, 59.275, 59.39 , 59.509, 59.628, 59.746, 59.864, 59.982,
        60.1  , 60.218, 60.335, 60.453, 60.571])
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
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786, 20.786,
        20.786, 20.786, 20.786, 20.786, 20.786])
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

