import numpy as np
import matplotlib.pyplot as plt

from . import transmission_spectrum as trs


class Sightline(object):

	def __init__(self, redshift, wl0=700., wlf=1250., include_low_density=True, include_high_density=True, include_cgm=True):
		"""
		Simulate an IGM sightline for the parameters given in Steidel et al. 2018

		source_redshift - the redshift of the sourve
		include_cgm - whether to also calculate the CGM contribution
		"""

		self.zs = redshift
		self.include_cgm = include_cgm
		self.include_low_density = include_low_density
		self.include_high_density = include_high_density

		# paramters for the cloud distributions:
		self.logA_igm_ld = 9.305
		self.logA_igm_hd = 7.542
		self.logA_cgm = 6.716

		self.beta_igm_ld = 1.635
		self.beta_igm_hd = 1.473
		self.beta_cgm = 1.381

		self.wl0 = wl0
		self.wlf = wlf

		# the redshift above which the CGM must be accounted for (within 700 km/s of
		# the galaxy redshift, see Steidel et al. 2018) 
		self.cgm_redshift =  self.zs - ((700. / 3.e5) * (1. + self.zs))

		# the wavelegnth grid on which to calculate the cloud rest-frame
		# optical depths:
		self.wl = np.arange(self.wl0 * (1. + self.zs), 
			self.wlf * (1. + self.zs), 0.1)
		self.wl_rest = self.wl / (1. + self.zs)

		# the final transmission spectrum:
		self.transmission = np.ones_like(self.wl, dtype=float)

		# the intervals in density and redshift:
		self.dlnhi = 0.1
		self.dz = 1.e-5

		# the cloud densities to draw from in the mcmc simulations:
		self.ld_lhni_vals = np.arange(12.0, 15.2, self.dlnhi)
		self.hd_lnhi_vals = np.arange(15.2, 21.0, self.dlnhi)
		self.cgm_lnhi_vals = np.arange(13.0, 21.0, self.dlnhi)

		# the redshift values to draw from in the mcmc simulations:
		self.ld_z_vals = np.arange(0.0, self.zs, self.dz)
		self.hd_z_vals = np.arange(0.0, self.zs, self.dz)
		self.cgm_z_vals = np.arange(self.cgm_redshift, self.zs, 1.e-7)

		# the pdfs for the cloud densities (will be used to sample from):
		self.hd_nhi_pdf = self._set_high_density_nhi_pdf()
		self.ld_nhi_pdf = self._set_low_density_nhi_pdf()
		self.cgm_nhi_pdf = self._set_cgm_nhi_pdf()

		# the pdfs for the redshifts (will be used to sample from):
		self.hd_z_pdf = self._set_high_density_z_pdf()
		self.ld_z_pdf = self._set_low_density_z_pdf()
		self.cgm_z_pdf = self._set_cgm_z_pdf()
		# calculate the absolute numbers of each type of cloud:
		self.ntot_ld = 10 ** self.logA_igm_ld *  self._density_integral(beta=self.beta_igm_ld, x1=10**15.2, x2=10**12.0) *\
			self._z_integral(gamma=2.5, z1=self.zs, z2=0.0)
		self.ntot_hd = 10 ** self.logA_igm_hd * self._density_integral(beta=self.beta_igm_hd, x1=10**21.0, x2=10**15.2) *\
			self._z_integral(gamma=1.0, z1=self.zs, z2=0.0)
		self.ntot_cgm = 10 ** self.logA_cgm * self._density_integral(beta=self.beta_cgm, x1=10**21.0, x2=10**13.0) *\
			self._z_integral(gamma=1.0, z1=self.zs, z2=self.cgm_redshift)

		self.ntot_total = self.ntot_ld + self.ntot_hd + self.ntot_cgm


	def _density_integral(self, beta, x1, x2):

		_f = 1 - beta
		return ((x1**_f)/_f) - ((x2**_f)/_f)


	def _z_integral(self, gamma, z1, z2):

		_g = gamma + 1
		return (((z1+1)**_g)/_g) - (((z2+1)**_g)/_g)


	def _set_low_density_nhi_pdf(self):

		num = np.empty_like(self.ld_lhni_vals)

		# total normalization factor:
		norm_fac = 10 ** self.logA_igm_ld * self._z_integral(gamma=2.5, z1=self.zs, z2=0.0)

		for i, _n in enumerate(self.ld_lhni_vals):
			num[i] = norm_fac * self._density_integral(beta=self.beta_igm_ld, 
				x1=np.power(10, _n+self.dlnhi), x2=np.power(10, _n))

		return num/np.sum(num)


	def _set_low_density_z_pdf(self):

		num = np.empty_like(self.ld_z_vals)

		# total normalization factor:
		norm_fac = 10 ** self.logA_igm_ld * self._density_integral(beta=self.beta_igm_ld, x1=10**15.2, x2=10**12.0)

		for i, _z in enumerate(self.ld_z_vals):
			num[i] = norm_fac * self._z_integral(gamma=2.5, z1=_z, z2=_z+self.dz)

		return num/np.sum(num)


	def _set_high_density_nhi_pdf(self):

		num = np.empty_like(self.hd_lnhi_vals)

		# total normalization factor:
		norm_fac = 10 ** self.logA_igm_hd * self._z_integral(gamma=1.0, z1=self.zs, z2=0.0)

		for i, _n in enumerate(self.hd_lnhi_vals):
			num[i] = norm_fac * self._density_integral(beta=self.beta_igm_hd, x1=np.power(10, _n+0.1), x2=np.power(10, _n))


		return num/np.sum(num)


	def _set_high_density_z_pdf(self):

		num = np.empty_like(self.hd_z_vals)

		# total normalization factor:
		norm_fac = 10 ** self.logA_igm_hd * self._density_integral(beta=self.beta_igm_hd, x1=10**21.0, x2=10**15.2)

		for i, _z in enumerate(self.hd_z_vals):
			num[i] = norm_fac * self._z_integral(gamma=1.0, z1=_z, z2=_z+self.dz)

		return num/np.sum(num)


	def _set_cgm_nhi_pdf(self):

		num = np.empty_like(self.cgm_lnhi_vals)

		# total normalization factor:
		norm_fac = 10 ** self.logA_cgm * self._z_integral(gamma=1.0, z1=self.zs, z2=self.cgm_redshift)

		for i, _n in enumerate(self.cgm_lnhi_vals):
			num[i] = norm_fac * self._density_integral(beta=self.beta_cgm, x1=np.power(10, _n+0.1), x2=np.power(10, _n))


		return num/np.sum(num)


	def _set_cgm_z_pdf(self):

		num = np.empty_like(self.cgm_z_vals)

		# total normalization factor:
		norm_fac = 10 ** self.logA_cgm * self._density_integral(beta=self.beta_cgm, x1=10**21.0, x2=10**13.0)

		for i, _z in enumerate(self.cgm_z_vals):
			num[i] = norm_fac * self._z_integral(gamma=1.0, z1=_z, z2=_z+self.dz)

		return num/np.sum(num)


	def _sample_doppler_parameter(self, size):

		b = np.linspace(0.1, 200., 1000)
		b0 = 23.0
		hb = ((4. * b0 ** 4) / (b ** 5)) * np.exp(-1. * (b0 ** 4) / (b ** 4))

		hb_prob = hb / np.sum(hb)

		return np.random.choice(a=b, p=hb_prob, size=size)


	def generate_clouds(self, print_progress=True):

		self.cloud_redshifts = np.empty(shape=0)
		self.lnhi = np.empty(shape=0)
		self.doppler_params = np.empty(shape=0)

		# get low density parameters
		if self.include_low_density:
			self.ld_cloud_redshifts = np.random.choice(a=self.ld_z_vals, p=self.ld_z_pdf, size=int(self.ntot_ld))
			self.ld_could_lnhi = np.random.choice(a=self.ld_lhni_vals, p=self.ld_nhi_pdf, size=int(self.ntot_ld))
			self.cloud_redshifts = np.concatenate((self.cloud_redshifts, self.ld_cloud_redshifts))
			self.lnhi =  np.concatenate((self.lnhi, self.ld_could_lnhi))
			self.doppler_params = np.concatenate((self.doppler_params, 
				self._sample_doppler_parameter(size=int(self.ntot_ld))))

		self.nclouds_low_density = len(self.cloud_redshifts)

		# get high density parameters
		if self.include_high_density:
			self.hd_cloud_redshifts = np.random.choice(a=self.hd_z_vals, p=self.hd_z_pdf, size=int(self.ntot_hd))
			self.hd_cloud_lnhi = np.random.choice(a=self.hd_lnhi_vals, p=self.hd_nhi_pdf, size=int(self.ntot_hd))
			self.cloud_redshifts = np.concatenate((self.cloud_redshifts, self.hd_cloud_redshifts))
			self.lnhi =  np.concatenate((self.lnhi, self.hd_cloud_lnhi))
			self.doppler_params = np.concatenate((self.doppler_params, 
				self._sample_doppler_parameter(size=int(self.ntot_hd))))

		self.nclouds_high_density = len(self.cloud_redshifts) - self.nclouds_low_density

		# get cgm parameters
		if self.include_cgm:
			self.cgm_cloud_redshifts = np.random.choice(a=self.cgm_z_vals, p=self.cgm_z_pdf, size=int(self.ntot_cgm))
			self.cgm_cloud_lnhi =  np.random.choice(a=self.cgm_lnhi_vals, p=self.cgm_nhi_pdf, size=int(self.ntot_cgm))
			self.cloud_redshifts = np.concatenate((self.cloud_redshifts, self.cgm_cloud_redshifts))
			self.lnhi =  np.concatenate((self.lnhi, self.cgm_cloud_lnhi))
			self.doppler_params = np.concatenate((self.doppler_params, 
				self._sample_doppler_parameter(size=int(self.ntot_cgm))))

		self.nclouds_cgm = len(self.cloud_redshifts) - (self.nclouds_low_density + self.nclouds_high_density)
		self.nclouds = len(self.cloud_redshifts)


	def make_transmission_spectrum(self, print_progress=False):

		# only use the clouds of interest to the problem (redshift set
		# by the lower wavelegnth limit)
		zmin = self.zs - (1220. / self.wl0)
		ok = self.cloud_redshifts >= zmin

		taus = np.zeros_like(self.wl)
		self.t900 = []

		for i, (_z, _lnhi, _b) in enumerate(zip(self.cloud_redshifts[ok], 
			self.lnhi[ok], self.doppler_params[ok])):

			if print_progress:
				print('{:d}/{:d}'.format(i+1, len(self.cloud_redshifts[ok])))

			# get the transmission
			_wl = self.wl / (1. + _z)
			taus += trs.generate_transmission_spectrum(wl=_wl, nhi=10**_lnhi, b=_b)

		# now multuply spectrum by transmission
		self.transmission *= np.exp(-1. * taus)

		t900_mask = (self.wl_rest >= 880.) & (self.wl_rest <= 910.)
		self.t900 = np.median(self.transmission[t900_mask])


	def print_expected_number_of_clouds(self):

		print("\nNumber of low-density systems = {:.1f}".format(self.ntot_ld))
		print("Number of high-density systems = {:.1f}".format(self.ntot_hd))
		print("Number of cgm systems = {:.1f}".format(self.ntot_cgm))
		print("Total number of absorbers = {:.1f}".format(self.ntot_total))

		nabs_lls_hd = 10 ** self.logA_igm_hd *  self._density_integral(beta=self.beta_igm_hd, x1=10**21.0, x2=10**17.2) *\
			self._z_integral(gamma=1.0, z1=self.zs, z2=0.0)
		nabs_lls_cgm = 10 ** self.logA_cgm *  self._density_integral(beta=self.beta_cgm, x1=10**21.0, x2=10**17.2) *\
			self._z_integral(gamma=1.0, z1=self.zs, z2=self.cgm_redshift)

		print("\nNumber of LLS (high-density) = {:.1f}".format(nabs_lls_hd))
		print("Number of LLS (cgm) = {:.1f}".format(nabs_lls_cgm))
		print("Total number of LLS = {:.1f}\n".format(nabs_lls_hd + nabs_lls_cgm))


	def write_cloud_info_to_file(self, ofile):

		np.savetxt(fname=ofile, X=np.column_stack((self.cloud_redshifts, self.lnhi, self.doppler_params)),
			header=' z lnhi b')


	def plot_cgm_nhi_dist(self):

		# total normalization factor:
		norm_fac = 10 ** self.logA_cgm * self._z_integral(gamma=1.0, z1=self.zs, z2=self.cgm_redshift)

		_lnhi = np.linspace(13.0, 21.0, 1000)
		_nhi = np.power(10, _lnhi)
		_phi = 1 - self.beta_cgm

		_number = -1. * ((norm_fac * _nhi**_phi) / _phi)

		n_t_b1 = _number[0]

		fig, ax = plt.subplots(figsize=(6, 6))
		ax.minorticks_on()

		if self.cgm_cloud_lnhi.any():
			_h, _be = np.histogram(self.cgm_cloud_lnhi, bins=10)
			n_a_b1 = _h[0]
			if n_a_b1 <= 0.:
				n_a_b1 = 1.0
			ax.hist(self.cgm_cloud_lnhi, histtype='stepfilled', color='C0', lw=0.)
			ratio = n_t_b1/n_a_b1
		else:
			ratio = 1.0

		ax.plot(_lnhi, _number/ratio, color='k', lw=1.)

		plt.show()
		fig.clf()


	def plot_low_density_nhi_dist(self):

		# total normalization factor:
		norm_fac = 10 ** logA_igm_ld * self._z_integral(gamma=2.5, z1=self.zs, z2=0.0)

		_lnhi = np.linspace(12.0, 15.2, 1000)
		_nhi = np.power(10, _lnhi)
		_phi = 1 - self.beta_igm_ld

		_number = -1. * ((norm_fac * _nhi**_phi) / _phi)

		n_t_b1 = _number[0]

		fig, ax = plt.subplots(figsize=(6, 6))
		ax.minorticks_on()

		if self.ld_could_lnhi.any():
			_h, _be = np.histogram(self.ld_could_lnhi, bins=10)
			n_a_b1 = _h[0]
			if n_a_b1 <= 0.:
				n_a_b1 = 1.0
			ax.hist(self.ld_could_lnhi, bins=10, histtype='stepfilled', color='C0', lw=0.)
			ratio = n_t_b1/n_a_b1
		else:
			ratio = 1.0

		ax.plot(_lnhi, _number/ratio, color='k', lw=1.)

		plt.show()
		fig.clf()


	def plot_high_density_nhi_dist(self):

		# total normalization factor:
		norm_fac = 10 ** self.logA_igm_hd * self._z_integral(gamma=1.0, z1=self.zs, z2=0.0)

		_lnhi = np.linspace(15.2, 21.0, 1000)
		_nhi = np.power(10, _lnhi)
		_phi = 1 - self.beta_igm_hd

		_number = -1. * ((norm_fac * _nhi**_phi) / _phi)

		n_t_b1 = _number[0]

		fig, ax = plt.subplots(figsize=(6, 6))
		ax.minorticks_on()

		if self.hd_cloud_lnhi.any():
			_h, _be = np.histogram(self.hd_cloud_lnhi, bins=10)
			n_a_b1 = _h[0]
			if n_a_b1 <= 0.:
				n_a_b1 = 1.0
			ax.hist(self.hd_cloud_lnhi, bins=10, histtype='stepfilled', color='C0', lw=0.)
			ratio = n_t_b1/n_a_b1
		else:
			ratio = 1.0

		ax.plot(_lnhi, _number/ratio, color='k', lw=1.)

		plt.show()
		fig.clf()

	def plot_fnx_distrubution_function(self):
		"""
		Figure 31. from Steidel et al. 2018
		"""

		fig, ax = plt.subplots(figsize=(9, 9))
		ax.minorticks_on()

		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(20) 
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(20) 

		igm_data = np.array([[13.142867162909772, -11.473768115942031],
			[13.25204503011652, -11.65217391304348],
			[13.370187713160963, -11.768115942028986],
			[13.506610729722857, -11.971014492753625],
			[13.652195831908607, -12.231884057971016],
			[13.797846690689742, -12.536231884057973],
			[13.97972943352885, -12.797101449275363],
			[14.17077426199181, -13.115942028985508],
			[14.379924073051193, -13.405797101449277],
			[14.60731037989777, -13.753623188405799],
			[14.852998939126927, -14.202898550724639],
			[15.126064160902004, -14.753623188405799],
			[15.435251672409409, -15.188405797101451],
			[15.762653760838878, -15.666666666666668],
			[16.126528840842735, -16.26086956521739],
			[16.51771482679713, -16.91304347826087],
			[16.96306232848488, -17.3768115942029],
			[17.699843060925676, -18.53623188405797]])

		cgm_data = np.array([[13.517241379310343, -11.898305084745763],
			[13.662431941923774, -12.072639225181598],
			[13.816696914700543, -12.508474576271187],
			[13.989110707803992, -12.595641646489105],
			[14.17967332123412, -12.78450363196126],
			[14.379310344827585, -13.322033898305085],
			[14.61524500907441, -13.380145278450364],
			[14.860254083484573, -13.699757869249394],
			[15.141560798548094, -14.251815980629539],
			[15.441016333938293, -14.818401937046005],
			[15.76769509981851, -15.00726392251816],
			[16.130671506352087, -15.762711864406779],
			[16.52994555353902, -16.02421307506053],
			[16.96551724137931, -16.605326876513317]])


		for igmd in igm_data:
			ax.scatter(igmd[0], igmd[1], color='blue',
				marker='s', s=20)

		for cgmd in cgm_data:
			ax.scatter(cgmd[0], cgmd[1], color='red',
				marker='s', s=20)

		# plot the best fit relations:
		lnh_ld = np.linspace(12.0, 15.2, 1000)
		fnx_ld = np.power(10, self.logA_igm_ld) * np.power(10, lnh_ld)**-self.beta_igm_ld
		ax.plot(lnh_ld, np.log10(fnx_ld), color='blue', lw=1.)

		lnh_hd = np.linspace(15.2, 21.0, 1000)
		fnx_hd = np.power(10, self.logA_igm_hd) * np.power(10, lnh_hd)**-self.beta_igm_hd
		ax.plot(lnh_hd, np.log10(fnx_hd), color='black', lw=1.)

		lnh_cgm = np.linspace(13.0, 21.0, 1000)
		fnx_cgm = np.power(10, self.logA_cgm) * np.power(10, lnh_cgm)**-self.beta_cgm
		ax.plot(lnh_cgm, np.log10(fnx_cgm), color='red', lw=1.)

		ax.set_xlim(13.0, 18.0)
		ax.set_ylim(-18.9, -11.1)

		ax.set_ylabel(r'f(N,X)', fontsize=25)
		ax.set_xlabel(r'log (N$_{\mathrm{HI}}$)', fontsize=25)

		plt.show()
		fig.clf()

