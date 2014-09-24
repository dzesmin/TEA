
# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
#
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Lead scientist and coder Jasmina Blecic, 
# assistant coder Oliver M. Bowman.   
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

# ============== This file read the TEA config file, TEA.cfg ============== #

# name of the configuration file
cfg_name = 'TEA.cfg'

# Check if config file exists
try:
    f = open(cfg_name)
except IOError:
    print "\nConfig file is missing. Place the config file in the working directory\n"

# open config file
config = ConfigParser.RawConfigParser({})
config.read(cfg_name)

# read TEA section
maxiter      = config.getint('TEA', 'maxiter')
save_headers = config.getboolean('TEA', 'save_headers')
save_outputs = config.getboolean('TEA', 'save_outputs')
doprint      = config.getboolean('TEA', 'doprint')
times        = config.getboolean('TEA', 'times')
location_TEA = config.get('TEA', 'location_TEA')

# read PRE-ATM section
pre_atm_name   = config.get('PRE-ATM', 'pre_atm_name')
input_elem     = config.get('PRE-ATM', 'input_elem')
output_species = config.get('PRE-ATM', 'output_species')


