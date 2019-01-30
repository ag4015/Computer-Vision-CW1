import os
import sys

desc_filename = str(sys.argv[1])


# Get data descriptors from Caltech using MATLAB
os.system("cd RF_2019; matlab -nodesktop -nosplash -r \"run(\'init.m\');clc;get_descriptors(\'CalTech\', \'" + desc_filename + "\', 0);exit;\"")
# os.system('cd ..')
