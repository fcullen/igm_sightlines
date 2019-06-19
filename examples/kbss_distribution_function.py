import sys
sys.path.append('../')
import igm_sightlines as igm
import matplotlib.pyplot as plt

import numpy as np

import time

if __name__ == "__main__":

	# set the redshift
	z = 3.5

	# make a sightlines object
	sl = igm.kbss.Sightline(redshift=z, include_low_density=True, 
		include_high_density=True, include_cgm=True, wl0=700., wlf=1300.)
	
	# generate clouds for a random sightlines
	sl.generate_clouds(print_progress=False)

	# now plot the Steidel KBSS distribution function f(N,X)
	# against the one generated:
	sl.plot_fnx_distrubution_function()