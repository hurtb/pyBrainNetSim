import os
from optparse import OptionParser
from shutil import copyfile

fpath = os.path.dirname(os.path.abspath(__file__))

parser = OptionParser()
parser.add_option("-d", "--simdir", dest="sim_directory", help="Main Simulation Director", metavar="SDIR")

# Setup
(options, args) = parser.parse_args()
sim_directory = options.sim_directory
base_setting_file = os.path.join(fpath,"../generators/settings/base.py")
script_file = os.path.join(fpath, '../simulation/_simulation_script.py')

#1. Create internal folder structure
if not os.path.exists(sim_directory):
    os.makedirs(sim_directory )
if not os.path.exists(os.path.join(sim_directory, 'data')):
    os.makedirs(os.path.join(sim_directory, 'data'))


#2. Create base.py file
SETTINGS_FILE = 'settings.py'
if not os.path.exists(os.path.join(sim_directory, SETTINGS_FILE)):
    copyfile(base_setting_file, os.path.join(sim_directory, SETTINGS_FILE))
    # with open(os.path.join(sim_directory, SETTINGS_FILE), 'w+') as f:
    #     f.writelines(open(base_setting_file).readlines())

#3. Create simulate.py file
SIM_NAME = 'simulate.py'
if not os.path.exists(os.path.join(sim_directory, SIM_NAME)):
    copyfile(script_file, os.path.join(sim_directory, SIM_NAME))
    # with open(os.path.join(sim_directory, SIM_NAME), 'w') as f:
    #     f.write("ENVIRONMENT_ORIGIN = (-10, 10)")
    #     f.write("ENVIRONMENT_ORIGIN = (-10, 10)")
    #     f.write("ENVIRONMENT_ORIGIN = (-10, 10)")