# This file is open-source software under the GNU GENERAL PUBLIC LICENSE
# Version 2, June 1991

import os
from pathlib import Path
import sys
import numpy as np

ROOT = str(Path(__file__).parents[1]) + os.path.sep
sys.path.append(ROOT)
import tea.janaf as janaf
import tea.thermal_properties as tea


def test_setup_janaf_network_missing_species():
    molecules = 'Ti Ti+ TiO TiO2 TiO+'.split()
    janaf_data = janaf.setup_network(molecules)

    expected_stoich_vals = np.array([
        [ 0,  1,  0],
        [ 0,  1, -1],
        [ 1,  1,  0],
        [ 2,  1,  0]
    ])

    assert len(janaf_data) == 5
    np.testing.assert_equal(janaf_data[0], ['Ti', 'Ti+', 'TiO', 'TiO2'])
    np.testing.assert_equal(janaf_data[1], ['O', 'Ti', 'e'])
    np.testing.assert_equal(janaf_data[4], expected_stoich_vals)


def test_heat_capacity_single_temp():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    heat_capacity = janaf_data[2]
    temperature = 1500.0
    cp = tea.thermo_eval(temperature, heat_capacity)

    expected_cp = np.array([
        5.6636252 , 10.41029396,  4.23563153,  7.02137982,  8.00580904,
        4.19064967,  3.88455652,  6.65454913,  3.95900511,  2.49998117,
        2.49998117,  2.5033488 ,  2.49998117,  2.50707724])
    np.testing.assert_allclose(cp, expected_cp)


def test_heat_capacity_temp_array():
    molecules = 'H2O CH4 CO C He'.split()
    janaf_data = janaf.setup_network(molecules)
    heat_capacity = janaf_data[2]
    temperatures = np.arange(100.0, 4501.0, 200.0)
    cp = tea.thermo_eval(temperatures, heat_capacity)

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


def test_gibbs_free_energy_temp_array():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    gibbs_funcs = janaf_data[3]
    temperatures = np.arange(100.0, 4101.0, 500.0)
    gibbs = tea.thermo_eval(temperatures, gibbs_funcs)

    expected_gibbs = np.array([
        [-3.17133424e+02, -1.16088681e+02, -1.59818988e+02,
         -5.02592674e+02, -8.20487182e+01, -2.61580345e+01,
         -1.86912862e+01,  1.34767459e+02,  2.15155216e+01,
          2.46172614e+02, -1.73952313e+01,  8.40705686e+02,
          5.47846591e+02,  2.77901183e+02],
        [-7.19942299e+01, -3.83538117e+01, -4.66207721e+01,
         -1.05556070e+02, -3.32912275e+01, -2.37361101e+01,
         -1.64041870e+01,  1.90286902e+00, -1.49815655e+01,
          2.94108805e+01, -1.56631891e+01,  1.24152943e+02,
          7.58228197e+01,  3.00677680e+01],
        [-5.16120930e+01, -3.38239976e+01, -3.79394442e+01,
         -7.17936083e+01, -3.11258185e+01, -2.51133488e+01,
         -1.77460657e+01, -1.23649275e+01, -1.98927195e+01,
          8.59717728e+00, -1.66138218e+01,  5.79016593e+01,
          3.18033564e+01,  6.39222419e+00],
        [-4.47129619e+01, -3.34225599e+01, -3.52753255e+01,
         -6.01006070e+01, -3.13341056e+01, -2.62131192e+01,
         -1.87902703e+01, -1.86146276e+01, -2.22838364e+01,
          4.23524064e-01, -1.73388235e+01,  3.26889136e+01,
          1.49277086e+01, -2.85681422e+00],
        [-4.15327188e+01, -3.40086693e+01, -3.42015976e+01,
         -5.45090464e+01, -3.20565745e+01, -2.71075847e+01,
         -1.96337403e+01, -2.23952165e+01, -2.38381927e+01,
         -4.04703875e+00, -1.79077118e+01,  1.92925792e+01,
          5.89877483e+00, -7.89133267e+00],
        [-3.98624680e+01, -3.48977825e+01, -3.37428163e+01,
         -5.14044786e+01, -3.29091716e+01, -2.78585653e+01,
         -2.03446702e+01, -2.50470695e+01, -2.49885061e+01,
         -6.91376381e+00, -1.83734063e+01,  1.09318319e+01,
          2.26972437e-01, -1.11052554e+01],
        [-3.89344931e+01, -3.58722619e+01, -3.35705245e+01,
         -4.95326757e+01, -3.37755811e+01, -2.85049090e+01,
         -2.09616674e+01, -2.70695166e+01, -2.59028825e+01,
         -8.93367922e+00, -1.87669375e+01,  5.18817306e+00,
         -3.69351747e+00, -1.33607688e+01],
        [-3.84158187e+01, -3.68512157e+01, -3.35468397e+01,
         -4.83493264e+01, -3.46155197e+01, -2.90721134e+01,
         -2.15080647e+01, -2.86960551e+01, -2.66629713e+01,
         -1.04488680e+01, -1.91073082e+01,  9.81302145e-01,
         -6.58180440e+00, -1.50464658e+01],
        [-3.81405228e+01, -3.78024079e+01, -3.36057563e+01,
         -4.75832286e+01, -3.54147523e+01, -2.95772573e+01,
         -2.19996178e+01, -3.00526088e+01, -2.73143300e+01,
         -1.16368930e+01, -1.94072675e+01, -2.24469576e+00,
         -8.80940438e+00, -1.63642684e+01]])
    np.testing.assert_allclose(gibbs, expected_gibbs)


def test_tea_network_init():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 H2 H C O'.split()
    tea_net = tea.Tea_Network(pressure, temperature, molecules)

    expected_stoich_vals = np.array([
        [0, 2, 1],
        [1, 4, 0],
        [1, 0, 1],
        [1, 0, 2],
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]])

    np.testing.assert_equal(tea_net.pressure, pressure)
    np.testing.assert_equal(tea_net.temperature, temperature)
    np.testing.assert_equal(tea_net.input_species, molecules)
    np.testing.assert_equal(tea_net.species, molecules)
    np.testing.assert_equal(tea_net.elements, ['C', 'H', 'O'])


def test_tea_network_cp_default_temp():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    tea_net = tea.Tea_Network(pressure, temperature, molecules)
    cp = tea_net.heat_capacity()

    expected_cp = np.array([
        5.26408044, 9.48143057, 4.11030773, 6.77638503, 7.34238673,
        4.05594463, 3.72748083, 6.3275286 , 3.79892261, 2.49998117,
        2.49998117, 2.50082308, 2.49998117, 2.51092596])
    assert np.shape(cp) == (nlayers, len(tea_net.species))
    np.testing.assert_allclose(cp[0], expected_cp)


def test_tea_network_cp_input_temp():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    tea_net = tea.Tea_Network(pressure, temperature, molecules)

    temps = [100.0, 600.0, 1200.0]
    cp = tea_net.heat_capacity(temps)

    expected_cp = np.array([
       [4.00494915, 4.00001798, 3.50040662, 3.51291495, 4.00314507,
        3.50040662, 3.38614788, 3.50798378, 3.92412613, 2.49998117,
        2.49998117, 2.55831326, 2.49998117, 2.85081563],
       [4.3688933 , 6.28146429, 3.6614513 , 5.69140811, 5.44749578,
        3.62140061, 3.52722736, 5.26865079, 3.55128183, 2.49998117,
        2.49998117, 2.50154471, 2.49998117, 2.54063323],
       [5.26408044, 9.48143057, 4.11030773, 6.77638503, 7.34238673,
        4.05594463, 3.72748083, 6.3275286 , 3.79892261, 2.49998117,
        2.49998117, 2.50082308, 2.49998117, 2.51092596]])

    assert np.shape(cp) == (len(temps), len(tea_net.species))
    np.testing.assert_allclose(cp, expected_cp)
    np.testing.assert_equal(tea_net.temperature, temperature)


def test_tea_network_gibbs_default_temp():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    tea_net = tea.Tea_Network(pressure, temperature, molecules)
    gibbs = tea_net.gibbs_free_energy()

    expected_gibbs = np.array([
        -49.70275118, -33.59076581, -37.17544326, -68.58699428,
        -31.08382889, -25.35365299, -17.97578591, -13.94856633,
        -20.48067821,   6.44982554, -16.77498672,  51.21052551,
         27.33544072,   3.95818325])
    assert np.shape(gibbs) == (nlayers, len(tea_net.species))
    np.testing.assert_allclose(gibbs[0], expected_gibbs)

