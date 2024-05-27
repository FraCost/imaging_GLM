import numpy as np

neural_data_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\GLM\\neural_data.csv' 
behav_data_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\GLM\\behavioral_data.csv' 
coord_data_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\GLM\\rois_coord_S1.csv'

paws = paws = ['FR', 'HR', 'FL', 'HL']
color_paws = ['red', 'magenta', 'blue', 'cyan']
trials = np.arange(1, 26+1)
blocks = np.array([[1, 6], [7, 16], [17, 26]])
