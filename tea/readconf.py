
############################# BEGIN FRONTMATTER ################################ 
#                                                                              # 
#   TEA - calculates Thermochemical Equilibrium Abundances of chemical species #
#                                                                              #
#   TEA is part of the PhD dissertation work of Dr. Jasmina                    #
#   Blecic, who developed it with coding assistance from                       #
#   undergraduate M. Oliver Bowman and under the advice of                     #
#   Prof. Joseph Harrington at the University of Central Florida,              #
#   Orlando, Florida, USA.                                                     #
#                                                                              #
#   Copyright (C) 2014-2016 University of Central Florida                      #
#                                                                              #
#   This program is reproducible-research software: you can                    #
#   redistribute it and/or modify it under the terms of the                    #
#   Reproducible Research Software License as published by                     #
#   Prof. Joseph Harrington at the University of Central Florida,              #
#   either version 0.3 of the License, or (at your option) any later           #
#   version.                                                                   #
#                                                                              #
#   This program is distributed in the hope that it will be useful,            #
#   but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#   Reproducible Research Software License for more details.                   #
#                                                                              #
#   You should have received a copy of the Reproducible Research               #
#   Software License along with this program.  If not, see                     #
#   <http://planets.ucf.edu/resources/reproducible/>.  The license's           #
#   preamble explains the situation, concepts, and reasons surrounding         #
#   reproducible research, and answers some common questions.                  #
#                                                                              #
#   This project was started with the support of the NASA Earth and            #
#   Space Science Fellowship Program, grant NNX12AL83H, held by                #
#   Jasmina Blecic, Principal Investigator Joseph Harrington, and the          #
#   NASA Science Mission Directorate Planetary Atmospheres Program,            #
#   grant NNX12AI69G.                                                          #
#                                                                              #
#   See the file ACKNOWLEDGING in the top-level TEA directory for              #
#   instructions on how to acknowledge TEA in publications.                    #
#                                                                              #
#   We welcome your feedback, but do not guarantee support.                    #
#   Many questions are answered in the TEA forums:                             #
#                                                                              #
#   https://physics.ucf.edu/mailman/listinfo/tea-user                          #
#   https://physics.ucf.edu/mailman/listinfo/tea-devel                         #
#                                                                              #
#   Visit our Github site:                                                     #
#                                                                              #
#   https://github.com/dzesmin/TEA/                                            #
#                                                                              #
#   Reach us directly at:                                                      #
#                                                                              #
#   Jasmina Blecic <jasmina@physics.ucf.edu>                                   #
#   Joseph Harrington <jh@physics.ucf.edu>                                     #
#                                                                              #
############################## END FRONTMATTER #################################

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


