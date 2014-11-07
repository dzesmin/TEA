
# ******************************* START LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances under thermochemical equilibrium conditions.
#
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Project developers included graduate student 
# Jasmina Blecic and undergraduate M. Oliver Bowman. 
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

import ConfigParser
import os

# =============================================================================
# This code reads the TEA config file, TEA.cfg. There are two sections in 
# TEA.cfg file: the TEA section and the PRE-ATM section. The TEA section 
# carries parameters and booleans to run and debug TEA. The PRE-ATM section
# carries paramaters to make pre-atmospheric file. 
# =============================================================================

# Get current working directory
cwd = os.getcwd() + '/'

# Name of the configuration file
cfg_name = cwd + 'TEA.cfg'

# Check if config file exists
try:
    f = open(cfg_name)
except IOError:
    print("\nConfig file is missing. Place TEA.cfg in the working directory\n")

# open config file
config = ConfigParser.RawConfigParser({})
config.read(cfg_name)

# read TEA section
maxiter      = config.getint    ('TEA', 'maxiter')
save_headers = config.getboolean('TEA', 'save_headers')
save_outputs = config.getboolean('TEA', 'save_outputs')
doprint      = config.getboolean('TEA', 'doprint')
times        = config.getboolean('TEA', 'times')
location_TEA = config.get       ('TEA', 'location_TEA')
abun_file    = config.get       ('TEA', 'abun_file')
location_out = config.get       ('TEA', 'location_out')

# read PRE-ATM section
PT_file        = config.get('PRE-ATM', 'PT_file')
pre_atm_name   = config.get('PRE-ATM', 'pre_atm_name')
input_elem     = config.get('PRE-ATM', 'input_elem')
output_species = config.get('PRE-ATM', 'output_species')


