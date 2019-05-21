import numpy as np
import numba
from numba import jit

@jit(nopython=True)
def voigt_x(wl, wl_i, b):

	# see section 2.1 of Tepper Garcia 2006
	# note b is in km/sec
	# lam and lam_i in Angstrom

	c = 2.99792458e5 # km/s
	delta_lambda_d = (b / c) * wl_i
	return (wl - wl_i) / delta_lambda_d


@jit(nopython=True)
def voigt_a(wl_i, b, gamma_i):

	# see section 2.1 of Tepper Garcia 2006
	# note b is in km/sec
	# lam and lam_i in Angstrom
	# Gamma_i is Einstein A?

	c = 2.99792458e5 # km/s
	wl_km = wl_i * 1.0e-13
	delta_lambda_d = (b / c) * wl_i  # angstrom
	return wl_i * (wl_km) * gamma_i/ (4 * np.pi * c * delta_lambda_d)


@jit(nopython=True)
def voigt_ca(wl_i, b, f_i):

	# see section 2.1 of Tepper Garcia 2006
	# and equation 10 of Inoue and Iwata 2008
	# note b is in km/sec
	# wl_i in Angstrom

	li = wl_i * 1.e-8 # to cm
	c = 2.99792458e10 # cm/s
	m_e = 9.1095e-28 # grams
	e_e = 4.8032e-10 # esu
	A = np.sqrt(np.pi) * e_e * e_e * f_i
	delta_lambda_d = (b * 1.0e5/c) * li
	B = m_e * c * c * delta_lambda_d
	return A*li*li/B


@jit(nopython=True)
def voigt_H(x, a):

	# see footnote 4 of Tepper Garcia 2006
	# lim x->0 H1(x) -> -2/sqrt(pi)
	# lim x->\infty H1(x) -> 0

	x2 = x * x
	H0 = np.exp(-1.*x2)
	Q  = 1.5/x2
	A = a/np.sqrt(np.pi)/x2 * (H0*H0*(4*x2*x2 + 7*x2 + 4 + Q) - Q -1.)
	return H0 - A


@jit(nopython=True)
def sig_i(wl, wl_i, b, f_i, gamma_i):

	a = voigt_a(wl_i=wl_i, b=b, gamma_i=gamma_i)
	Ca = voigt_ca(wl_i=wl_i, b=b, f_i=f_i)
	x  = voigt_x(wl=wl, wl_i=wl_i, b=b)
	H  = voigt_H(x=x, a=a);

	return Ca * H;


@jit(nopython=True)
def generate_transmission_spectrum(wl, nhi, b):

	sigma_ls = np.zeros_like(wl)

	line_params =  [(1215.67, 4.6986e8, 4.1641e-1), 
					(1025.72, 5.5751e7, 7.9142e-2), 
					(972.537, 1.2785e7, 2.9006e-2), 
					(949.743, 4.1250e6, 1.3945e-2),
					(937.803, 1.6440e6, 7.8035e-3), 
					(930.748, 7.5684e5, 4.8164e-3), 
					(926.226, 3.8694e5, 3.1850e-3), 
					(923.150, 2.1425e5, 2.2172e-3),
					(920.963, 1.2631e5, 1.6062e-3), 
					(919.352, 7.8340e4, 1.2011e-3), 
					(918.128, 5.0659e4, 9.2190e-4), 
					(917.181, 3.3927e4, 7.2310e-4),
					(916.429, 2.3409e4, 5.7769e-4), 
					(915.824, 1.6572e4, 4.6886e-4), 
					(915.329, 1.1997e4, 3.8577e-4), 
					(914.919, 8.8574e3, 3.2124e-4),
					(914.576, 6.6540e3, 2.7035e-4), 
					(914.286, 5.0767e3, 2.2967e-4), 
					(914.039, 3.9276e3, 1.9677e-4), 
					(914.826, 3.0769e3, 1.6987e-4),
					(913.641, 2.4380e3, 1.4767e-4), 
					(913.480, 1.9519e3, 1.2917e-4), 
					(913.339, 1.5776e3, 1.1364e-4), 
					(913.215, 1.2862e3, 1.0051e-4),
					(913.104, 1.0571e3, 8.9321e-5), 
					(913.006, 8.7524e2, 7.9736e-5), 
					(912.918, 7.2967e2, 7.1476e-5), 
					(912.839, 6.1221e2, 6.4319e-5),
					(912.768, 5.1673e2, 5.8087e-5), 
					(912.703, 4.3857e2, 5.2635e-5), 
					(912.645, 3.7418e2, 4.7845e-5), 
					(912.592, 3.2081e2, 4.3619e-5),
					(912.543, 2.7631e2, 3.9877e-5), 
					(912.499, 2.3903e2, 3.6551e-5), 
					(912.458, 2.0762e2, 3.3585e-5), 
					(912.420, 1.8103e2, 3.0931e-5),
					(912.385, 1.5843e2, 2.8550e-5), 
					(912.353, 1.3913e2, 2.6407e-5), 
					(912.325, 1.2258e2, 2.4474e-5)]

	for lp in line_params:

		_sig_x = sig_i(wl=wl, wl_i=lp[0], b=b, f_i=lp[2], gamma_i=lp[1])
		sigma_ls += _sig_x

	# add the lyman continuum absorption:
	for i in range(wl.shape[0]):
		if wl[i] <= 912.0:
			sigma_ls[i] += 6.3e-18 * (wl[i]/912.) ** 2.75

	return nhi * sigma_ls