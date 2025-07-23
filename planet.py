# SPONCHpop project - lite
# Planet Formation Module
#
# by Anna Thomas
# 2023-2025
"""
Created on Wed Aug 23 11:55:44 2023

@author: annathomas
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from tqdm import tqdm

# local modules
import disk as ds

# =============================================================================
# planet class
# =============================================================================

class planet_disk_evo:
    """
    A class that represents a planet. It simulates the evolution of the planet by updating its attributes at each time step.
    
    Attributes:
        PDM (bool):                 switch for planetesimal driven migration
        migration (bool):           switch for type I and II migration
        peb_acc (bool):             switch for pebble accretion
        plan_acc (bool):            switch for planetesimal accretion
        gas_acc (bool):             switch for gas accretion (both envelope contraction and runaway gas accretion)
        grain_growth (bool):        switch for grain growth
        pebble_drift (bool):        switch for pebble drift

        disk (object):              disk model
        m_0 (float):                birth mass of the embryo, kg
        a_0 (float):                inital semi-major axis of the embryo, m
        t_0 (float):                birth time of the planet, yrs
        B (float):                  width of the feeding zone in hill spheres, dimensionless
        p_ratio (float):            factor for planetesimal accretion into atmosphere/core, dimensionless
        dt_yrs (float):             time step for the simulation, yrs

        m (float):                  mass of the planet at timestep, kg
        a (float):                  semi-major axis of the planet from host star at timestep, m

        t_final (float):            final time for the simulation, yrs
        dt (float):                 time step for the simulation, s
        t_range (np.ndarray):       array of time values for the simulation

        key_vals (np.ndarray):      array to store key values at each time step
                                    [0] label, [1] m, [2] a, [3] dmdt, [4] dadt, [5] menv, [6] mcore, [7] r, [8] peb_to_pln_ratio, [9] dmdt_peb, [10] dmdt_pln, [11] dmdt_gas
        oth_vals (np.ndarray):      array to store other values at each time step.
                                    [0] e_orb (0), [1] i_orb (0), [2] J_rot (0 kg m^2 s^-1), [3] J_rot_plane (0 degrees)

        ind_peb (int):              index of the time step when the planet reaches pebble isolation mass.
        ind_con (int):              index of the time step when the planet stops Kelvin-Helmholtz contraction.
        ind_mig (int):              index of the time step when the planet changes migration regime from type I to type II

        cum_peb_mass (float):       cumulative mass of pebbles accreted by the planet
        cum_pln_mass (float):       cumulative mass of planetesimals accreted by the planet

        pebble_evo (np.ndarray):    array to store the evolution of pebble surface density at each time step.
    """

    def __init__(self, m_0, a_0, t_0, B, p_ratio, dt_yrs, PDM, migration, peb_acc, plan_acc, gas_acc, grain_growth, pebble_drift, disk):
        # birth mass, birth semi major axis, birth time, width of feeding zone, factor for pebble isolation, disk, ratio of planetesimals accreted to core/envelope mass, process switches
        self.PDM            = PDM
        self.migration      = migration
        self.peb_acc        = peb_acc
        self.plan_acc       = plan_acc
        self.gas_acc        = gas_acc
        self.grain_growth   = grain_growth
        self.pebble_drift   = pebble_drift

        self.disk           = disk                                                          # disk model       
        self.m_0            = m_0                                                           # kg, inital mass of embryo
        self.t_0            = t_0                                                           # yrs, birth time of planet
        self.a_0            = a_0                                                           # m, initial distance of embryo to central star
        self.B              = B                                                             # dimensionless, width of feeding zone in hill spheres
        self.p_ratio        = p_ratio                                                       # dimensionless, fraction of planetsimals adding to the mass of the envelope/core after peb_iso is reached
        self.dt_yrs         = dt_yrs                                                        # yrs, time step of simulation
        
        self.m              = m_0                                                           # kg, initialising mass of planet
        self.a              = a_0                                                           # m, initialising distance of planet to central star
        
        self.t_final        = self.t_0 + 3e6                                                # yrs, final time of simulation
        self.dt             = self.dt_yrs * ds.yr                                           # s, time step of simulation
        self.t_range        = np.arange(float(0.0), self.t_final, self.dt_yrs)*ds.yr        # array of time values for full disk evolution and planet formation
        
        self.key_vals       = np.zeros([(len(self.t_range)) , 12])                          # array to store key values at each time step
        self.oth_vals       = np.zeros([(len(self.t_range)) , 4])                           # array to store other values at each time step
        
        self.key_vals[0,1]  = self.m_0                                                      # initialising mass of planet
        self.key_vals[0,2]  = self.a_0                                                      # initialising semi major axis of planet
        
        self.ind_peb        = None                                                          # index of the time step when the planet reaches pebble isolation mass
        self.ind_con        = None                                                          # index of the time step when the planet stops envelope contraction
        self.ind_mig        = None                                                          # index of the time step when the planet changes migration regime from type I to type II

        self.cum_peb_mass   = 0
        self.cum_pln_mass   = 0

        self.rad_grid                       = np.array(self.disk.rstruct())                 # radial grid of disk
        self.grid_sizes                     = np.diff(self.rad_grid)                        # grid widths
        self.r_grid_centres                 = 0.5*(self.rad_grid[1:] + self.rad_grid[:-1])  # locations of grid centres
        self.planetesimal_surface_density   = [self.disk.get_disk_plan(radius = r, status = self.disk.validate( radius = r , time = 0.0 )) for r in self.r_grid_centres]   # planetesimal surface density at each grid centre
        self.planetesimal_grid              = np.vstack((self.rad_grid[:-1], self.planetesimal_surface_density)).T                                                              # planetesimal grid
        self.pebble_surface_density         = self.disk.get_pebsig(radius = self.r_grid_centres , time = 0.0, status=1)
        self.gas_surface_density            = self.disk.get_Sigma(radius = self.r_grid_centres , time = 0.0, status=1) 
        self.dust_surface_density           = self.disk.get_Sigma(radius = self.r_grid_centres , time = 0.0, status=1) * (self.disk.dustr * self.disk.dtgr)
        self.pebble_sigma_evo               = np.zeros((len(self.t_range), len(self.r_grid_centres)))

    def eq11(self, eta, rcap, vkep, temperature, radius):  
        eta =  (self.disk.get_cs(temperature)/vkep)**2
        return vkep * max(eta , ( (3 * rcap) / (2*radius) ) )
    
        """
        Calculates relative velocity between pebbles and planet

        Args:
            eta (float):         eta = (cs/vkep)^2, dimensionless
            rcap (float):        capture radius of of planet for pebbles, m
            vkep (float):        keplerian velocity of planet, m/s
            temperature (float): temperature of the disk midplane at location of planet, K
            radius (float):      semi major axis of planet, m

        Returns:
            float: relative velocity between pebbles and planet, m/s
        """
    
    def single_planet_case(self):
        """
        Simulates the evolution of a single planet in a disk, updating its attributes at each time step.
        """
        dmdt_pln                = 0
        self.st                 = self.disk.get_st_ini(time=0.0, radius=self.r_grid_centres, status=1)
        self.gas_surface_density            = self.disk.get_Sigma(radius = self.r_grid_centres , time = float(0.0), status = 1).flatten()
        self.dust_surface_density           = self.disk.get_Sigma(radius = self.r_grid_centres , time = float(0.0), status = 1).flatten() * (self.disk.dustr * self.disk.dtgr)
        dust_and_pebbles                    = (self.dust_surface_density + self.pebble_surface_density).flatten()
        
        for i, t in enumerate(self.t_range):
            if i == 0:
                progress_bar = tqdm(total=len(self.t_range), desc="Progress", unit="step")
            progress_bar.update(1)
            if i == len(self.t_range) - 1:
                progress_bar.close()  

            validation           = self.disk.validate(radius=self.a, time=self.t_range[i])
            if validation != 1:
                print('Radius/ time out of bounds')
                break

            if self.t_range[i] <= self.t_0*ds.yr:
                self.key_vals[i,:]                  = 0
                
                self.gas_surface_density            = self.disk.get_Sigma(radius = self.r_grid_centres , time = self.t_range[i], status=1).flatten()
                self.dust_surface_density           = self.disk.get_Sigma(radius = self.r_grid_centres , time = self.t_range[i], status=1).flatten() * (self.disk.dustr * self.disk.dtgr)
                dust_and_pebbles                    = (self.dust_surface_density + self.pebble_surface_density).flatten()
                temp_array                          = self.disk.get_T(radius=self.r_grid_centres, time=self.t_range[i], status=validation)
                
                if self.grain_growth == True:
                    self.st          = self.disk.stokes_calculator(radius=self.r_grid_centres, time=self.t_range[i], st=self.st, status=validation)       # dimensionless, stokes number across disk at each grid centre                                                                                                    # dimensionless, stokes number at location of planet
                elif self.grain_growth == False:
                    self.st          = np.minimum(self.disk.get_St_turb(temperature=temp_array), self.disk.get_St_drft(radius=self.r_grid_centres, temperature=temp_array))

                if self.pebble_drift == True:
                    radial_drift                        = self.disk.get_vr(radius = self.r_grid_centres, time = self.t_range[i], St = self.st, status = validation)
                    twodee_in                           = self.pebble_surface_density * radial_drift
                    twodee_out                          = np.roll(twodee_in, -1)
                    twodee_out[-1]                      = 0
                    delta_sigma                         = (-twodee_in + twodee_out) / self.grid_sizes
                    self.pebble_surface_density        += delta_sigma * self.dt
                elif self.pebble_drift == False:
                    radial_drift                        = np.zeros_like(self.r_grid_centres)
                    prev_gas_surface_density            = self.disk.get_Sigma(radius = self.r_grid_centres , time = self.t_range[i-1], status=1).flatten()
                    self.pebble_surface_density        *= self.gas_surface_density / prev_gas_surface_density
                    
                self.pebble_sigma_evo[i, :]         = self.pebble_surface_density.copy()
            
            elif self.t_range[i] >= self.t_0*ds.yr:
                self.key_vals[i,1]   = self.m                                                                            # kg, mass of planet
                self.key_vals[i,2]   = self.a    
                bindex               = np.digitize(self.a, self.r_grid_centres) 

                fkep                 = self.disk.get_Omega(radius=self.a)                                                     # /s, keplerian frequency
                vkep                 = fkep * np.pi * 2 * self.a                                                              # m/s, keplarian velocity
                
                r                    = self.disk.get_mr(mass=self.m)                                                          # m, radius of planet
                self.key_vals[i,7]   = r

                temp_array           = self.disk.get_T(radius=self.r_grid_centres, time=self.t_range[i], status=validation)                            # K, tempurature of disk midplane at semi-major axis of planet (np.array)
                temp                 = temp_array[bindex]                                                                                                      # K, temperature of disk midplane at location of planet

                if self.grain_growth == True:
                    self.st          = self.disk.stokes_calculator(radius=self.r_grid_centres, time=self.t_range[i], st=self.st, status=validation)       # dimensionless, stokes number across disk at each grid centre
                    stokes           = self.st[bindex]                                                                                                          # dimensionless, stokes number at location of planet
                elif self.grain_growth == False:
                    self.st          = np.minimum(self.disk.get_St_turb(temperature=temp_array), self.disk.get_St_drft(radius=self.r_grid_centres, temperature=temp_array))
                    stokes           = np.minimum(self.disk.get_St_turb(temperature=temp), self.disk.get_St_drft(radius=self.a, temperature=temp))

                hillrad              = self.disk.get_hr(mass=self.m, radius=self.a, status=validation)                                      # m, hill radius of planet
                eta                  = self.disk.get_eta(temperature=temp, vkep=vkep)                     # dimensionless, eta = (cs/vkep)^2

                peb_iso              = self.disk.get_iso_peb(radius=self.a, temperature=temp, status=validation)                         # kg, eq 12, john chambers 2018
                
                def eq8(rcap): 
                    vrel             = self.eq11(eta=eta, rcap=rcap, vkep=vkep, temperature=temp, radius=self.a)
                    return (rcap/hillrad)**3 + (((2 * self.a * vrel ) / (3 * hillrad * vkep)) * (rcap/hillrad)**2 ) - (8 * stokes)
                
                rcap_g               = hillrad                                                                           # initial guess for planet's pebble capture radius          
                rcap_peb             = fsolve(eq8, rcap_g)[0]                                                            # m, capture radius of planet for pebbles
                vrel                 = self.eq11(eta=eta, rcap=rcap_peb, vkep=vkep, temperature=temp, radius=self.a)                                      # m/s, relative velocity between pebbles and planet   

                fg                   = self.disk.get_fg(mass=self.m, r_plan=r, vrel=self.eq11(eta=eta, rcap=hillrad, vkep=vkep, temperature=temp, radius=self.a))          # gravitational focussing factor of planet
                rcap_plan            = r * fg**0.5                                                                       # m, capture radius of planet for planetesimals, chambers 2014

                self.gas_surface_density            = self.disk.get_Sigma(radius = self.r_grid_centres , time = self.t_range[i], status=1).flatten()
                self.dust_surface_density           = self.disk.get_Sigma(radius = self.r_grid_centres , time = self.t_range[i], status=1).flatten() * (self.disk.dustr * self.disk.dtgr)
                dust_and_pebbles                    = (self.dust_surface_density + self.pebble_surface_density).flatten()

                #   PLANETESIMAL ACCRETION TAKES PLACE UNTIL DEPELETION OF PLANETESIMALS, REGARDLESS OF PEBBLE ISOLATION MASS BEING REACHED
                if self.plan_acc == True:
                    modded_grid, modified_sigma = self.disk.update_plan_grid(input_s=self.a, planetesimal_grid=self.planetesimal_grid, B=self.B, m_dot=dmdt_pln, mass=self.m, dt=self.dt, status=validation)
                    dmdt_pln                    = self.disk.get_plnacc(sig_plan=modified_sigma, radius=self.a, rcap_plan=rcap_plan, fg=fg)
                elif self.plan_acc == False:
                    dmdt_pln                    = 0

                #   BELOW PEBBLE ISOLATION MASS - PEBBLE ACCRETION UNTIL ISOLATION MASS IS REACHED
                if (self.m < peb_iso) and self.a > (0.03 * ds.auSI):
                    dmdt_gas                = 0 
                    self.ind_peb            = i
                    m_env                   = (self.key_vals[self.ind_peb,1] * 0.1)                                       # kg, mass of gas envelope, assumed to be 10% of planet mass SOURCE
                    self.key_vals[i,5]      = m_env
                    m_core                  = (self.key_vals[self.ind_peb,1] * 0.9)                                       # kg, mass of planet core, assumed to be 90% of planet mass
                    self.key_vals[i,6]      = m_core
                    self.cum_pln_mass       += dmdt_pln * self.dt

                    if self.peb_acc == True:
                        dmdt_peb             = self.disk.get_pebacc(rcap_peb = rcap_peb, radius= self.a, vrel = vrel, temperature = temp, st = stokes, sigma = self.pebble_surface_density[bindex])
                        pebble_deduct        = (dmdt_peb * self.dt) / (self.grid_sizes[bindex] * 2 * np.pi * self.a)
                        pebble_deduct        = np.minimum(pebble_deduct, self.pebble_surface_density[bindex]) 
                        self.cum_peb_mass   += np.minimum(dmdt_peb * self.dt, (self.pebble_surface_density[bindex] * 2 * np.pi * self.a * self.grid_sizes[bindex]))

                    elif self.peb_acc == False:
                        dmdt_peb                            = 0
                        pebble_deduct                       = 0
                    
                    self.key_vals[i,3]           = dmdt_peb + dmdt_pln + dmdt_gas
                    self.key_vals[i,9]           = dmdt_peb
                    self.key_vals[i,10]          = dmdt_pln
                    self.key_vals[i,11]          = dmdt_gas
                    self.m                      += (dmdt_peb + dmdt_gas + dmdt_pln) * self.dt 

                #   ABOVE PEBBLE ISOLATION MASS - ENVELOPE CONTRACTION AND POSSIBLE RUNAWAY GAS ACCRETION
                if (self.m >= peb_iso) and self.a > (0.03 * ds.auSI):
                    m_core                       = (self.key_vals[self.ind_peb,1] * 0.9) + ((1-self.p_ratio) * dmdt_pln * self.dt)
                    self.key_vals[i,6]           = m_core
                    dmdt_peb                     = 0

                    if self.gas_acc == True: 
                        if m_core > m_env:                                                                                              # if mass of planet core exceeds mass of gas envelope, envelope contraction
                            self.ind_con         = i
                            dmdt_gas             = self.disk.get_envcon(mcore = m_core, menv = m_env, temperature = temp, status = validation)
                        elif m_core <= m_env:                                                                                           # if mass of planet gas envelope exceeds mass of core, runaway gas accretion
                            dmdt_gas             = self.disk.get_gasacc(time=self.t_range[i], radius=self.a, mass=self.m, temperature=temp, status=validation)
                    elif self.gas_acc == False:
                        dmdt_gas = 0

                    m_env                       += (dmdt_gas * self.dt) + (dmdt_pln * self.dt * self.p_ratio)
                    self.key_vals[i,5]           = m_env
                    self.key_vals[i,3]           = dmdt_peb + dmdt_pln + dmdt_gas
                    self.m                       = m_core + m_env
                    self.key_vals[i,9]           = dmdt_peb
                    self.key_vals[i,10]          = dmdt_pln * self.p_ratio
                    self.key_vals[i,11]          = dmdt_gas

                #   MIGRATION
                if self.migration == True:
                    dadt                  = self.disk.get_mig(time=self.t_range[i], radius=self.a, mass=self.m, temperature=temp, status=validation)
                    if hillrad < self.disk.get_scaleheight(radius=self.a, temperature=temp):
                        dadt                  = self.disk.get_mig(time=self.t_range[i], radius=self.a, mass=self.m, temperature=temp, status=validation)
                    if hillrad < self.disk.get_scaleheight(radius=self.a, temperature=temp):
                        self.ind_mig      = i
                    if self.PDM == True:
                        dadt_PDM          = (self.disk.get_PDM(radius=self.a, sigma=modified_sigma, mass=self.m, bh_pdm=2.2, mass_horseshoe=1 * ds.M_e, status=validation)) * -ds.auSI/ds.yr
                    elif self.PDM == False:
                        dadt_PDM          = 0
                elif self.migration == False:
                    dadt                  = 0
                    dadt_PDM              = 0
                if self.a < (0.03 * ds.auSI):       # if planet is within 0.03 AU of central star, migration and accretion stops
                    dadt                  = 0
                    dadt_PDM              = 0
                    dmdt_gas              = 0
                    dmdt_peb              = 0
                    dmdt_pln              = 0
                    self.key_vals[i,5]    = max(self.key_vals[:,5])
                    self.key_vals[i,6]    = max(self.key_vals[:,6])

                if self.cum_pln_mass > 0:
                    peb_to_pln_ratio = self.cum_peb_mass / self.cum_pln_mass
                else:
                    peb_to_pln_ratio = float('inf')

                if self.pebble_drift == True:
                    radial_drift                        = self.disk.get_vr(radius=self.r_grid_centres, time=self.t_range[i], St=self.st, status=validation)
                    twodee_in                           = self.pebble_surface_density * radial_drift
                    twodee_out                          = np.roll(twodee_in, -1)
                    twodee_out[-1]                      = 0
                    self.pebble_surface_density[bindex] = max(0, (self.pebble_surface_density[bindex] - pebble_deduct))
                    delta_sigma                         = (-twodee_in + twodee_out) / self.r_grid_centres
                    self.pebble_surface_density        += delta_sigma * self.dt
                    self.pebble_sigma_evo[i, :]         = self.pebble_surface_density.copy()
                elif self.pebble_drift == False:
                    radial_drift                    = np.zeros_like(self.r_grid_centres)
                    twodee_in                           = self.pebble_surface_density * radial_drift
                    twodee_out                          = np.roll(twodee_in, -1)
                    twodee_out[-1]                      = 0
                    delta_sigma                         = (-twodee_in + twodee_out) / self.r_grid_centres
                    prev_gas_surface_density            = self.disk.get_Sigma(radius = self.r_grid_centres , time = self.t_range[i-1], status=1).flatten()
                    self.pebble_surface_density        *= self.gas_surface_density / prev_gas_surface_density
                    self.pebble_surface_density[bindex] = max(0, (self.pebble_surface_density[bindex] - pebble_deduct))
                    self.pebble_sigma_evo[i, :]         = self.pebble_surface_density.copy()
                
                self.cum_peb_mass                  += dmdt_peb * self.dt

                self.key_vals[i,8]                  = peb_to_pln_ratio
                self.key_vals[i,4]                  = dadt + dadt_PDM
                self.a                             += self.key_vals[i,4] * self.dt

    def get_vals(self):
        return self.key_vals, self.t_range

    def get_other(self):
        return self.ind_peb, self.ind_con, self.ind_mig, self.pebble_surface_density, self.planetesimal_grid, self.pebble_sigma_evo