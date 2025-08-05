# CHNOPS project
# Chemical model
#
# by Mihkel Kama and Oliver Shorttle
# 2020

debug	= False
import matplotlib.pyplot as plt
import numpy
import numpy as np
import physcon
import disk as ds

# print("")
# print("----------------------")
# print("----------------------")
# print("-- Chemistry module --")
# print("----------------------")
# print("----------------------")
# print("   All units are SI unless stated otherwise.")
# print("")

# print("-- All units are assumed SI unless explicitly stated otherwise.")
# TODO: Standard values and constants that should really be defined somewhere else, like the main.py module (or the chemistry module as global variables)
year_in_seconds		= ds.yr
time100yr			= 100.0 * year_in_seconds
desorbtime			= time100yr
Nsites				= 2e+19	# number density of binding sites
P_atmo 				= 1.01325 # Pa per 1 atmospheric pressure
cm3_to_m3			= (1./100.)**3
Pa_to_Ba 			= (1e+5/1e+4) # pressure units from SI to cgs: 1 N = 1e+5 dyn; 1 m^2 = 1e+4 cm^2
auSI 				= 1.498e+11 # AU in meters
P_atmo 				= 1.01325 # Pa per 1 atmospheric pressure
m_H2 				= 2 * physcon.m_p      # mass of an H2 molecule in kg
m_S 				= 32 * physcon.m_p    # mass of an H2S molecule in kg
m_H2S 				= m_H2 + m_S    # mass of an H2S molecule in kg
m_Fe 				= 56 * physcon.m_p    # mass of an Fe atom in kg
m_FeS 				= m_Fe + m_S    # mass of an FeS unit in kg
m_H 				= 1.67e-27   	  # mass of an H atom in kg

M_sun			= 1.989e+30	            # kg
R_sun			= 6.957e+8	            # m
M_e             = ds.M_e                # kg, earth mass
R_e             = 6.3781e6              # m, earth radius
M_j             = 1.898e27              # kg, jupiter mass
gamma			= 1.4		            # adiabatic index
auSI			= 1.496e+11             # m
yr          	= 365.2425 * 24 * 3600  # s


# returns the evaporation time in seconds
# based on Eqs. (4.30, 4.31) in Tielens (2005)
def sublimetime( EdesorbK, massamu, Tdust ):
		EbindJ		= EdesorbK * physcon.k_B
		nu_z		= numpy.sqrt( 2*Nsites*EbindJ / ( numpy.pi**2 * massamu * physcon.m_p ) )
		time_evap	= numpy.exp( EdesorbK / Tdust ) / nu_z
		return time_evap

# sublimation temperature in K
def getTsub( EdesorbK, massamu ):
		EbindJ		= EdesorbK * physcon.k_B
		nu_z		= numpy.sqrt( 2*Nsites*EbindJ / ( numpy.pi**2 * massamu * physcon.m_p ) )
		Tsub		= EdesorbK / numpy.log( desorbtime * nu_z )
		return Tsub

# Class for defining species
class Species:
	# print("-- Species are defined by these parameters:")
	# print("   name, mass (amu), desorption energy Edes (K), Tsub (K), Cabun (C/H), Nabun, Oabun, Pabun, Sabun")
	# print("   Note that Tsub will be calculated from Edes if Tsub<0")
	# print("!!!!!")
	# print("Eventually, the aim is to input the stoichiometry and corresponding relative abundances of a species as lists of some kind (e.g., 'Fe:S' and '0.5:1' if half of all Fe and all S were locked in FeS, and total Fe/H = 2*S/H. Parse these inside the class.")
	# print("!!!!!")
	def __init__( self, name, mass, Edes, Tsub, Cabun, Nabun, Oabun, Pabun, Sabun ):
		self.name	= str(name)
		self.mass	= float(mass)
		self.Edes	= Edes
		if Tsub > 0:
			self.Tsub	= Tsub
		else:
			self.Tsub	= getTsub( Edes, mass )
		self.Cabun	= float(Cabun)
		self.Nabun	= float(Nabun)
		self.Oabun	= float(Oabun)
		self.Pabun	= float(Pabun)
		self.Sabun	= float(Sabun)

def loadchem_sponchbob():
	# Load up the chemical reservoirs to consider
	import os
	# Get the directory where this chemistry module is located
	chem_dir = os.path.dirname(os.path.abspath(__file__))
	dat_file_path = os.path.join(chem_dir, 'sponchbob.dat')
	rawspecies	= numpy.loadtxt(dat_file_path,delimiter='	',comments='#', \
							dtype={'names': ('name', 'mass', 'Edes', 'Tsub', 'C/H', 'N/H', 'O/H', 'P/H', 'S/H'), \
							'formats': ('S6', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')})
	specieslist	= {}
	for elem in rawspecies:
		specieslist[elem[0]]	= Species( elem[0], elem[1], elem[2], elem[3], elem[4], elem[5], elem[6], elem[7], elem[8] )
	if debug:
		for key in specieslist.keys():
			print(" %s	%.1f K" % (specieslist[key].name,specieslist[key].Tsub))
	return specieslist

# 'microchem' is the bare-basics chemical kinetics model, which attempts to capture only the most critical processes and timescales for each element
def getrates_microchem( n_gas, T_kin, x_CO, x_N2, ionrate_He=1e-16 ):
	'''
	getrates_microchem
	Inputs (in SI units):
	n_gas		m^-3; the gas density
	T_kin		K; the local kinetic temperature
	x_CO		n_CO/n_H_tot the CO abundance
	x_N2		n_CO/n_H_tot the N2 abundance
	ionrate_He	s^-1 the He ionization rate by CR-like particles
	'''
	rates	= {}
	# E_a notes:
	# H2S to FeS and the reverse are from Lauretta et al. (1996), listed there in J/mole, so division by R=8.314 J/(K*mole) converts to K
	E_a		= { 'unit':'K', 'H2StoFeS': -27950.0/physcon.R, 'FeStoH2S':-92610.0/physcon.R }
	cm3_to_m3	= (1./100.)**3
	
	######################################
	# CO and N2 reaction with He+
	# following Bergin et al. (2014, FaDi)
	######################################
	x_He		= 0.28	# the elemental He abundance (n_He_tot/n_H_tot)
	k_H2_1		= 7.2e-15 * cm3_to_m3	# m^3/s the rate coefficient for H2 + He+ into H2+ (Landenslager et al. 1974)
	k_H2_2		= 3.7e-14 * numpy.exp(-35.0/T_kin) * cm3_to_m3	# m^3/s the rate coefficient for H2 + He+ into H + H+ (Landenslager et al. 1974) 
	k_H2		= k_H2_1 + k_H2_2
	k_CO		= 1.6e-9 * cm3_to_m3	# m^3/s the rate coefficient for CO + He+ (Barlow 1984 PhD thesis; UMIST)
	k_N2		= 9.6e-10 * cm3_to_m3	# m^3/s the rate coefficient for N2 + He+ (UMIST)
	n_Heplus	= x_He*ionrate_He * ( x_CO*k_CO + x_N2*k_N2 + k_H2 )**(-1)
	n_H2		= n_gas * (1.0-x_He)/2.0	# the approximate number density of H2

	reformation_factor_CO_Hep = 0.0		# fraction of CO destroyed by He+ which reforms into CO before C goes anywhere else
	rates['CO']	= (1-reformation_factor_CO_Hep) * x_CO * n_H2 * k_CO * n_Heplus


	###############################################
	# Sulfur conversion from volatile to refractory
	###############################################
	a_Fe	= 1e-5	# radius=a_Fe=10um; pure Fe grains
	delta_gasFe	= 100*10
	#
	# S: H2S to FeS conversion, assuming unlimited availability of pure Fe grains
	rho_Fe_SI		= 7.874e+3 # kg/m^3
	n_Fe_SI			= rho_Fe_SI / (55.845*physcon.m_p) # atoms/m^3
	areavol_Fe		= 2.3*physcon.m_p * (delta_gasFe*rho_Fe_SI)**(-1) * (n_gas/a_Fe)		# total Fe grains surface area per unit volume m^2/m^3; see Mihkel's research notebook 2021.08.10
	rates['H2S']	= - 5.6 * numpy.exp( E_a['H2StoFeS'] / T_kin ) * (0.01/3600.0) * areavol_Fe * n_Fe_SI
	rates['FeS']	= - rates['H2S']

	# This is just to sanity-check the numbers
	if False:
		print("-- Grain area per volume = ", areavol_Fe, " m^2/m^3")
		print("   Rate of H2S destruction = ", rates['H2S'], " r/(m^3*s)")
		print("   Years to destroy local gas-phase H2S is approximately = ", n_gas*1e-5/(rates['H2S']*year_in_seconds), " yr" )
	#	Lauretta et al. (1996, Icarus);
	#	*0.01/3600 converts from cm/h to m/s
	#	/n_Fe_SI applies the density of pure Fe to get reactions reactions/m^2/s
	#	*areavol_Fe multiplies by Fe grain area per unit volume m^2/m^3 to get reactions/(m^3*s)

	return rates

# A quick sanity check of the numbers for the H2S-to-FeS conversion rate per m^3 per s
if False:
	TTT = numpy.arange(1000)+1
	RRR = numpy.array( [ getrates_microchem( 1e+20, float(tempT) )['H2S'] for tempT in TTT ] )
	plt.plot( TTT, RRR, 'k-' )
	plt.ylim([1e-70,1e-40])
	plt.yscale('log')
	plt.show()

def get_rate_coefficient_microchem_S( reaction, T_kin ):

	'''
	get_rate_coefficient_microchem_S
	Returns the rate coefficient for a specific reaction in the sponchpop microchem network for volatile/refractory sulfur conversion
	Inputs:		reaction - string, e.g. "H2StoFeS" or "FeStoH2S"; specifies the reaction
				T_kin - float [K]; kinetic temperature at which to evaluate the rate coefficient
	Outputs:	rate coefficient [ g cm^-2 s^-1 Pa^-1 ]
	Notes:
	H2S + Fe_solid -> FeS_solid + H2 reaction rate coefficient data from Lauretta et al. (1996xyz)
	''' 
	E_a		= { 'unit':'K', 'H2StoFeS': -27950.0/physcon.R, 'FeStoH2S':-92610.0/physcon.R }
	if reaction == 'H2StoFeS':
		return 5.6 * numpy.exp( E_a['H2StoFeS'] / T_kin ) / ( 60.0 * P_atmo )  # [ g cm^-2 s^-1 Pa^-1 ]   division by 60 sec/h as the coefficients by Lauretta are per hour
	elif reaction == 'FeStoH2S':
		return 10.3 * numpy.exp( E_a['FeStoH2S'] / T_kin ) / ( 60.0 * P_atmo ) # [ g cm^-2 s^-1 Pa^-1 ]   division by 60 sec/h as the coefficients by Lauretta are per hour

# propagate_chem_microchem_S updated function to include accretion_removal
# -- I HAVE ADDED THE COMMENTS BACK INTO THE FUNCTION FROM MK'S ORIGINAL FUNCTION
def propagate_chem_microchem_S(
    disk_object,
    sigma_gas_array,
    sigma_dust_array,
    absolute_time,
    chem_endtime,
    dtime,
    species_coldens,
    accretion_removal = None,
    global_start_time = 0,  # MK 20250717: obsolete parameter?
    store_all_coldens_timesteps=False,
    chem_abun_change_limit_factor=0.01): # MK 20250722 added the abundance cap factor

    '''	
	propagate_chem_microchem_S
	Calculates chemical reaction rates and uses them to move the chemistry forward by an amount total_time of time
	Inputs:	disk_object - disk object that will be used as the environment for the chemistry
			species_coldens - a dictionary containing the surface density arrays for each chemical species
			absolute_time - [s], time at the start of this timestep in the global (disk, planet) simulation
			chem_endtime - [s]; total time to run the chemical kinetics in this instance (can be e.g., equal to the length of one disk evolution or planet formation timestep)
			dtime - [s]; the chemistry timestep; this needs to be dtime <= chem_endtime and best if < chem_endtime/5
			global_start_time = 0 - absolute initial time that sets the reference gas and dust surface densities (used in scaling the column densities as the disk evolves)
			store_all_coldens_timesteps = False - Boolean; create an array that stores all column density values across the disk at each timestep?; intended mainly for diagnostic use; USE CAREFULLY: if =True then the risk is running out of virtual memory
            chem_abun_change_limit_factor = ... - maximum fraction of a chemical species that can be removed in a timestep; range [0,1]
	Outputs:
			species_coldens - updated surface density arrays for S, Fe, FeS
	'''
	# TODO for species_coldens: calculate exact mass fraction of Fe in total dust of solar composition
	# species_coldens = { 'H2S':numpy.array( [ sigma_gas_array*( 1e-5*m_H2S/m_H2 ) ] ), 
	# 					'FeS_dust':numpy.array( [ sigma_gas_array * 0 ] ), 
	# 					'Fe_dust':numpy.array( [ sigma_dust_array * 0.2 ] ) }
	# total_S_mass = sigma_gas_array * (1e-5 * m_S / m_H2)  # S mass per m^2

	# # Split S equally between H2S and FeS (by S atoms)
	# H2S_mass = 0.5 * total_S_mass * (m_H2S / m_S)      # convert S mass to H2S mass
	# FeS_mass = 0.5 * total_S_mass * (m_FeS / m_S)      # convert S mass to FeS mass
    if dtime > chem_endtime:
        print(" !! ERROR: the chemistry time step is larger than the disk or planet evolution time step; this violates global time conservation.")
    elif dtime > chem_endtime/5:
        if debug: print(" !! WARNING: the chemistry timestep is larger than 1/5 of the disk or planet evolution timestep; this may lead to a poor solution.")
    if not 0 <= chem_abun_change_limit_factor <= 1:
        print("!! ERROR: chem_abun_change_limit_factor=%.3e outside allowed range [0,1]" % chem_abun_change_limit_factor )
        raise SystemExit(0)
    
    #radius_array 		= np.linspace(0.8*auSI,1.2*auSI,10) # MK 20250717: temporarily commented this out for easier debugging: 
    radius_array        = disk_object.rstruct()
    a_grain 			= 1e-3 							# grain size in meters -- TODO: needs to be a general property for sponchpop dust (either global or local, but always callable)
    rho_grain 			= 1.25e+3 						# internal density of grains in kg m^-3 -- TODO: needs to be a general property for sponchpop dust (either global or local, but always callable)
    N_r 				= len( radius_array )

    # Secondary values, needed internally for the sulfur chemistry below; but might be good to have them as sponchpop-level properties:
    vol_grain 			= (4/3)*np.pi*a_grain**3
    mass_grain 			= rho_grain * vol_grain
    area_grain 			= 4*np.pi*a_grain**2
    numcol_dust_array 	= sigma_dust_array / mass_grain # the column number density of dust grains

    this_time = 0
    while this_time <= chem_endtime:
        if debug: 
            print(" -- chemistry subloop: this_time=%.2e s" % (this_time))
            print("    sigma_gas: ", sigma_gas_array )
        tkin_array = disk_object.tstruct(absolute_time + this_time)
        if debug: print("    tkin_array: ", tkin_array )
        for i_r in range(len(sigma_gas_array)):
			# Calculate the H2 and H2S partial pressures:
			# h_P not needed right now, but can be obtained from sponchpop: disk_object.get_scaleheight( radius_array, temperature_array )
			# h_P = c_s / disk_object.Omega # the pressure scaleheight
			### The partial pressure of H2 gas:
			# DONE: Tkin(r,t) from sponchpop and rho0->Sigma(r,t)
            P_H2 = physcon.k_B * (sigma_gas_array[i_r]/m_H2) * tkin_array[i_r]
            P_H2_cgs = P_H2 * Pa_to_Ba
			# TODO: H2 pressure actually needs to use H2 mass (not total); mu=2 is already used for the particle mass
			### The partial pressure of H2S gas:
			# TODO: H2S, Fe, FeS surface densities from sponchpop
            rho_h2s = species_coldens['H2S'][i_r]
            P_H2S = physcon.k_B * (rho_h2s/m_H2S) * tkin_array[i_r]
            P_H2S_cgs = P_H2S * Pa_to_Ba

			# Calculate the reaction rate dS_FeS_dt for FeS and the corresponding change of FeS mass per unit of grain area in one timestep:
			# DONE: T_kin from sponchpop
			# this_T_kin = tkin_array[i_r]
			# calculate dS_FeS / dt = kf*PH2S - kr*PH2
            dS_FeS_dt = get_rate_coefficient_microchem_S('H2StoFeS', tkin_array[i_r]) * P_H2S_cgs - get_rate_coefficient_microchem_S('FeStoH2S', tkin_array[i_r]) * P_H2_cgs  # [ g cm^-2 s^-1 ]
            dS_FeS_cgs = dS_FeS_dt * dtime # [ g cm^-2 ] cgs units; change in FeS mass per unit of grain surface area
            dS_FeS = dS_FeS_cgs * (1e+4/1e+3) # [ kg m^-2 ] SI units; change in FeS mass per unit of grain surface area

			# Dust area available per unit volume on the midplane:
			# TODO: dust and pebble surface densities..
			# TODO: ..and sizes from sponchpop 
            numvol_dust = numcol_dust_array[i_r]
            areavol_dust = numvol_dust * area_grain # total area of dust grains per unit volume [m^2 m^-3] -- note that actually it's per column, not per unit volume..
			# if FeS creation
            if dS_FeS > 0:
				# maximum number of FeS units (Fe:S=1:1) created in this time step is the lowest of however many S or Fe are available:
                max_num_FeS = min( species_coldens['Fe_dust'][i_r]/m_Fe, species_coldens['H2S'][i_r]/m_H2S ) * chem_abun_change_limit_factor # MK 20250722 added the abundance cap factor
				# mass of FeS created in this time step; make sure to not overshoot the amount of Fe or S or FeS available:
                dM_FeS = min(dS_FeS * areavol_dust, max_num_FeS * m_FeS) # [ kg m^-2 ]; change in FeS mass surface density in the disk
                # if dM_FeS < 0: 
                #     print(" !! ERROR: dM_FeS < 0 while dS_FeS > 0")
                dM_Fe = dM_FeS * (m_Fe / (m_Fe + m_S)) # [ kg m^-2 ]; change in Fe mass surface density in the disk
                dM_S = dM_FeS * (m_S / (m_Fe + m_S)) # [ kg m^-2 ]; change in S mass surface density in the disk
			# if FeS destruction:
            else:
				# because the destruction rate is negative, we use max() to find the smallest change in FeS mass:
                if debug: print("    check both dM_FeS are negative: ", -species_coldens['FeS_dust'][i_r], dS_FeS * areavol_dust )
                dM_FeS = max(-species_coldens['FeS_dust'][i_r]*chem_abun_change_limit_factor, dS_FeS * areavol_dust) # [ kg m^-2 ]; change in FeS mass surface density in the disk # MK 20250722 added the abundance cap factor
                if dM_FeS > 0: 
                     print(" !! ERROR: dM_FeS > 0 while dS_FeS < 0")
                dM_Fe = dM_FeS * (m_Fe / (m_Fe + m_S)) # [ kg m^-2 ]; change in Fe mass surface density in the disk
                dM_S = dM_FeS * (m_S / (m_Fe + m_S)) # [ kg m^-2 ]; change in S mass surface density in the disk
				# Skip chemical kinetics if temperature is between 400 K and 700 K
            if debug: print("    dS_FeS = %.3e,   i_r = %i,   dM_S = %.3e,   dM_FeS = %.3e" % (dS_FeS, i_r, dM_S, dM_FeS))
			# Update the radial surface density arrays for the relevant species (Fe, H2S, FeS) locally:
            species_coldens['FeS_dust'][i_r] += dM_FeS
            species_coldens['Fe_dust'][i_r] -= dM_Fe
            species_coldens['H2S'][i_r] -= ( dM_S + numpy.sign( dM_S ) * m_H2 )

            species_coldens['FeS_dust'][i_r] = max(0, species_coldens['FeS_dust'][i_r])  # prevent negative values
            species_coldens['Fe_dust'][i_r]  = max(0, species_coldens['Fe_dust'][i_r])    # prevent negative values
            species_coldens['H2S'][i_r]      = max(0, species_coldens['H2S'][i_r])      # prevent negative values
        if debug:
            print("    coldens FeS_dust: ", species_coldens['FeS_dust'] )
            print("    coldens Fe_dust: ", species_coldens['Fe_dust'] )
            print("    coldens H2S: ", species_coldens['H2S'] )
        this_time += dtime

    # --- remove FeS and Fe from species_coldens due to solid accretion by planet ---
    if accretion_removal is not None:
        for bindex, pebdeduct, pre_peb_sig in accretion_removal:
            if pre_peb_sig > 0:
                frac_removed = pebdeduct / pre_peb_sig
                species_coldens['FeS_dust'][bindex] -= frac_removed * species_coldens['FeS_dust'][bindex]
                species_coldens['Fe_dust'][bindex]  -= frac_removed * species_coldens['Fe_dust'][bindex]
			    # prevent negative values
                species_coldens['FeS_dust'][bindex] = max(0, species_coldens['FeS_dust'][bindex])
                species_coldens['Fe_dust'][bindex]  = max(0, species_coldens['Fe_dust'][bindex])

    return species_coldens