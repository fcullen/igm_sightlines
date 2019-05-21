import sys
sys.path.append('../')
from igm_sightlines import transmission_spectrum as trns
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# make a wavelength array
	wl = np.linspace(700, 1220., 1000)

	# get optical depth for a cloud with column density 10^17 cm^-2 and
	# doppler parameter 23 km/s
	tau = trns.generate_transmission_spectrum(wl=wl, nhi=10**17.0, b=23.0)

	# quick plot:
	plt.plot(wl, np.exp(-1. * tau))
	plt.ylim(0.0, 1.05)
	plt.show()