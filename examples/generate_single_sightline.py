import sys
sys.path.append('../')
import igm_sightlines as igm
import matplotlib.pyplot as plt

import numpy as np

import time

if __name__ == "__main__":

	make_spec = True

	# set the redshift
	z = 3.05

	# make a sightlines object
	sl = igm.kbss.Sightline(redshift=z, include_low_density=True, 
		include_high_density=True, include_cgm=True, wl0=700., wlf=1300.)

	# print expected number of clouds per sighline:
	sl.print_expected_number_of_clouds()
	
	# generate clouds for a random sightlines
	sl.generate_clouds(print_progress=False)

	# sanity check that the numbers generated are reasonable
	print("\nGenerated {:d} low density clouds".format(sl.nclouds_low_density))
	print("Generated {:d} high density clouds".format(sl.nclouds_high_density))
	print("Generated {:d} cgm clouds".format(sl.nclouds_cgm))
	print("Generated {:d} LSS clouds".format(len(sl.lnhi[sl.lnhi>=17.2])))
	print("Generated {:d} total clouds\n".format(sl.nclouds))

	if make_spec:

		# make the transmission spectrum for this sighline
		sl.make_transmission_spectrum(print_progress=True)
		
		# plot:
		fig, ax = plt.subplots(figsize=(15, 5))
		ax.minorticks_on()
		
		ax.plot(sl.wl/(1.+sl.zs), sl.transmission, ls='-', lw=0.5, color='k')

		ax.set_ylim(0.0, 1.05)
		ax.set_xlim(700, 1300)

		plt.show()
		plt.clf()
