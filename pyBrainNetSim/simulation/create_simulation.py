import os
from optparse import OptionParser

fpath = os.path.dirname(os.path.abspath(__file__))

parser = OptionParser()
parser.add_option("-d", "--simdir", dest="sim_directory", help="Main Simulation Director", metavar="SDIR")

# Setup
(options, args) = parser.parse_args()
sim_directory = options.sim_directory
base_setting_file = os.path.join(fpath,"../generators/random.py")

#1. Create internal folder structure
if not os.path.exists(sim_directory):
    os.makedirs(sim_directory )
if not os.path.exists(os.path.join(sim_directory, 'data')):
    os.makedirs(os.path.join(sim_directory, 'data'))


#2. Create base.py file
SETTINGS_FILE = 'base.py'
if not os.path.exists(os.path.join(sim_directory, SETTINGS_FILE)):
    with open(os.path.join(sim_directory, SETTINGS_FILE), 'w+') as f:
        f.writelines(open(base_setting_file).readlines())

#3. Create simulate.py file
SIM_NAME = 'simulate.py'
if not os.path.exists(os.path.join(sim_directory, SIM_NAME)):
    with open(os.path.join(sim_directory, SIM_NAME), 'w+') as f:
        f.write("")