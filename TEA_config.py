
# ******************************* END LICENSE *******************************
# Thermal Equilibrium Abundances (TEA), a code to calculate gaseous molecular
# abundances for hot-Jupiter atmospheres under thermochemical equilibrium
# conditions.
#
# This project was completed with the support of the NASA Earth and Space 
# Science Fellowship Program, grant NNX12AL83H, held by Jasmina Blecic, 
# PI Joseph Harrington. Lead scientist and coder Jasmina Blecic, 
# assistant coder for the first pre-release Oliver M. Bowman.  
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

import os

# ============================================================================= 
# Configuration file containing parameters and booleans to run or debug TEA.
# Change the parameters below to control how TEA runs. The default number of 
# iterations is the optimal value for common molecular species in hot Jupiters.
# ============================================================================= 

# ========    Sets maximum number of iteration   ======== 
maxiter      = 100      # (Def: 100)   Number of iterations the main pipeline
                        #              will run for each T-P point 

# ========          Controls output files        ======== 
save_headers = False    # (Def: False) Preserve headers for multi-T-P 
                        #              pre-atm files 
save_outputs = False    # (Def: False) Preserve intermediate outputs for 
                        #              multi-T-P pre-atm files

# ========   Controls debugging and tracking     ========
doprint      = False    # (Def: False) Enable various debug printouts 
times        = False    # (Def: False) Enable time printing for speed tests 


# ========         Tep and pressure file         ========
cwd     = os.getcwd()   # Get current directory name
tepfile = cwd + '/inputs/tepfile/HD209458b.tep'  # Tepfile


