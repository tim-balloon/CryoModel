#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as sp

from scipy.interpolate import interp1d
from scipy.integrate import simpson,trapz,quad

from radmodel import uprint
import mli_keller

#constants
#Stefan-Boltzmann constant
SIGMA = 5.6704E-12   #[J/s*cm^2*K^4]


class ThermalMaterial: 
    '''
    A class for materials used in the thermal modeling, useful quantities can be stored
    in this class
    '''
    def __init__(self, name, specificHeat=np.inf, conductivity=np.inf, latentHeat=np.inf ):
        '''
        conductivity[W/m/K]: thermal conductivity as a function of temp
        '''
        self.name = name
        self.specificHeat = specificHeat
        self.conductivity = conductivity
        self.latentHeat = latentHeat


    def getCond(self, T,):
        Ts = self.conductivity[:,0]
        ks = self.conductivity[:,1]

        return np.interp(T, Ts, ks)

    def getFlux(self,T1,T2,AOverL):
        '''
        calculates conductive flux
        '''

        Ts = self.conductivity[:,0]
        ks = self.conductivity[:,1]

        # use at least 200 points for the integral
        dT = np.min(Ts[1:] - Ts[:-1])/2
        N = int(np.max([(T2-T1)/dT, 200]))

        T  = np.linspace(T1, T2, N)

        return AOverL * simpson(np.interp(T, Ts, ks), x=T)
    

    def getFlux_quad(self, T1,T2,AOverL):
        '''
        calculates conductive flux, using quad
        '''

        Ts = self.conductivity[:,0]
        ks = self.conductivity[:,1]

        k_inte = interp1d(Ts, ks, kind='linear')
        
        return AOverL * quad(k_inte, T1, T2, epsrel=1e-2)[0]


folder = '/Users/frank/GoogleDrive/TIM/HK/CryoModel'


# Common materials
SS316 = ThermalMaterial(name='SS316', conductivity=np.loadtxt(folder+'/thermalProp/SS316_TC.csv'))
CF = ThermalMaterial(name='CF', conductivity=np.loadtxt(folder+'/thermalProp/CF_TC_Runyan.txt',))
G10 = ThermalMaterial(name='G10', conductivity=np.loadtxt(folder+'/thermalProp/g10_cv_combined.txt',))

NbTi = ThermalMaterial(name='NbTi', conductivity=np.loadtxt(folder+'/thermalProp/NbTi_cv.txt',))
PTFE = ThermalMaterial(name='PTFE', conductivity=np.loadtxt(folder+'/thermalProp/PTFE_cv.txt',))

Ag = ThermalMaterial(name='Silver', conductivity=np.loadtxt(folder+'/thermalProp/Ag_RRR30.txt',))

Manganin = ThermalMaterial(name='Manganin', conductivity=np.array(\
    [[0.1,0.4,1.,4.0,10.0,20.0,80.0,150.0,300.0],[0.006,0.02,0.06,0.5,2.0,3.3,13.0,16.0,22.0]]).T)


PhBr = ThermalMaterial(name='PhBr', conductivity=np.loadtxt(folder+'/thermalProp/PhBr_k.csv',))


class ThermalLink(object):
    def __init__(self, name, s1, s2, s3=None, flux=0):
        self.name = name
        self.s1   = s1
        self.s2   = s2
        self.s3   = s3
        self.flux = flux


class Radi(ThermalLink):
    def __init__(self, area=0, em=1.0, optical=0, **kwargs):
        super().__init__(**kwargs)
        self.type = 'Radiation'
        self.area = area
        self.em   = em
        self.op   = optical #optical load

    def calFlux(self):
        self.flux = self.area * self.em * SIGMA * (self.s1.T**4 - self.s2.T**4)+self.op
        self.s1.load -= self.flux
        self.s2.load += self.flux

        return self.flux


class MLI(ThermalLink):
    def __init__(self, area=0, em=0.15, N=20, N_s=20, P=1e-4, optical=0, **kwargs):
        super().__init__(**kwargs)
        self.type = 'MLI'
        self.area = area    # in m^2
        self.em   = em
        self.op   = optical #optical load
        self.NLayer = N     #number of layers
        self.N_s  = N_s     #layer density cm^-1
        self.P      = P

    def calFlux(self):
        self.flux = self.area*mli_keller.P_tot(self.P, self.NLayer, self.N_s,\
                                    self.s1.T, self.s2.T, e_r = self.em) +self.op
        self.s1.load -= self.flux
        self.s2.load += self.flux

        return self.flux

class Cond(ThermalLink):
    def __init__(self, AOverL=0, material=None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'Conduction'
        self.AOverL = AOverL
        self.material = material

    def calFlux(self):

        self.flux = self.material.getFlux(self.s2.T, self.s1.T, self.AOverL)
        self.s1.load -= self.flux
        self.s2.load += self.flux

        return self.flux

class Rod(Cond):
    '''
    A class created to simplify the calculation of A/L
    A/L will be calculated using D,L and N
    '''
    def __init__(self, D=0, L=0, N=1, material=None, **kwargs):
        super().__init__(**kwargs)
        self.type = 'Conduction'
        self.AOverL = (np.pi * D**2./4.) / L *N
        self.material = material


class Fridge(ThermalLink):
    '''
    This is a special link class that represents the Fridge.
    Useful quantities will be stored and calculated using this class.
    Noted that the flux(or load) calculated using calFlux method would not be
    added to the stages as loads on stages are specified to external loads.
    '''

    def __init__(self, stp1 = 10, stp2 = 8, stp3 = 2, eff=0.5, heaterE=5000, no_load_time=48*3600, recycle_time=2*3600,\
        FPU_heat_cap=2.74, **kwargs):
        '''
        stp1 [L]: Helium 4 volume (in stp liters) in the 1st stage(4He)
        stp2 [L]: Helium 3 volume (in stp liters) in the 2nd stage(3He-IC)
        stp3 [L]: Helium 3 volume (in stp liters) in the 3rd stage(3He-UC)
        eff: pumping efficiency
        heaterE [J]: Energy from the fridge heaters in one cycle
        no_load_time [s]: Hold time of the fridge still when there is no external loads
        recycle_time [s]: Time to cycle the fridge, i.e. time that still is not at operating temp
        FPU_heat_cap [J]: Energy needed to cool the cold element at the begining
        of each cycle(from 4k to 0.3k),

        Noted that the default value of 2.74J is estimated as: \
        2.74 = 2.5(two and half module)* 0.58(volume in L) * 2.7(Al density) * 0.7(Al Enthalpy[J/kg] at 4K)
        Here we are using a upper bound of the Al Enthalpy at 4K from: \
        https://arxiv.org/ftp/arxiv/papers/1501/1501.07100.pdf Fig.3
        '''

        super().__init__(**kwargs)
        self.type = 'Fridge'
        # total cooling power at the still in J,
        # He3 latent heat is around 25 J/mol at 0.3 K
        # He4 latent heat is around 85 J/mol at 1.0 K
        self.coldE1 = stp1/22.4*85 *eff
        self.coldE2 = stp2/22.4*25
        self.coldE3 = stp3/22.4*25
        self.heaterE = heaterE
        self.no_load_time = no_load_time

        self.para_load1 = self.coldE1/no_load_time  # still parasitic load in W
        self.para_load2 = self.coldE2/no_load_time  # still parasitic load in W
        self.para_load3 = self.coldE3/no_load_time  # still parasitic load in W

        self.FPU_heat_cap = FPU_heat_cap

        self.recycle_time = recycle_time

    def getStillLoad(self):
        self.still_load1 = self.s1.load + self.para_load1
        self.still_load2 = self.s2.load + self.para_load2
        self.still_load3 = self.s3.load + self.para_load3

        #print(self.still_load1,self.still_load2,self.still_load3)

        self.hold_time1 = (self.coldE1-self.FPU_heat_cap)/ self.still_load1
        self.hold_time2 = (self.coldE2)/ self.still_load2
        self.hold_time3 = (self.coldE3)/ self.still_load3

    def calFlux(self):
        self.getStillLoad()
        self.hold_time = np.min([self.hold_time1,self.hold_time2,self.hold_time3])
        self.cycle_flux = self.heaterE/self.hold_time

        return 0

class Source(ThermalLink):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'Source'

    def calFlux(self):
        self.s2.load += self.flux
        return self.flux



class Stages(object):
    def __init__(self, name, T0, Type=None,Wt=0):
        self.name = name
        self.T    = float(T0)
        self.Type = Type
        self.load = 0.
        self.Wt   = float(Wt)


class ThermalSystem(object):

    ### setting up the model
    def __init__(self, name,):
        self.name = name
        self.stages = []
        self.linkages = []

    def add_stage(self, stage):
        self.stages.append(stage)

    def add_stages(self, stages):
        for stage in stages:
            self.stages.append(stage)

    def add_linkage(self, link):
        self.linkages.append(link)

    ### print the details of the model
    def showLinks(self,):
        print('Linkages:')
        for link in self.linkages:
            if link.type == 'Conduction':
                print('  %-18s\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), \
                    '| %s\t'%link.type, '| %s, A/L: %3.2e m'%(link.material.name, link.AOverL))

            elif link.type == 'Radiation':
                print('  %-18s\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), \
                     '| %s\t'%link.type, '| area: %3.2e cm^2'%link.area)
            else:
                print('  %-18s\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), '|', link.type)

    def showStages(self,):
        print('Stages:')
        for stage in self.stages:
            print('  ', stage.name,', Temp: %1.2fK'%stage.T)

    def showSystem(self):
        print('System:', self.name)
        self.showStages()
        self.showLinks()

    def showLoads(self):
        print('Fluxes and Loadings on stages:')
        for stage in self.stages[1:]:
            print('  On %s Stage:'%stage.name,'')
            for link in self.linkages:
                if link.s2==stage and link.type != 'Fridge':
                    print('    Link: %-18s'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), '| Flux: ', uprint(link.flux, fmt='%6.3f') )
            # print('    -----------\n','   Total Loading on this stage: \t\t', uprint(stage.load, fmt='%6.3f'),'\n' )
            print('    -----------\n','   Net Loading on this stage: \t\t', uprint(stage.load, fmt='%6.3f'),'\n' )


    def showHoldtime(self, V=275):
        tankload = self.stages[-1].load
        dayuse = (power2mdot(tankload)/0.125*86400) #in LHe Liters
        holdtime = V/dayuse
        print('Daily LHe cost:',  end="\t")
        print("%1.2f L"%dayuse)

        print('Estimated hold time:',  end="\t")
        print("%1.2f days"%holdtime)



    def getLoads(self):
        stageFlux = []
        for stage in self.stages[1:]:
            fluxes = []
            for link in self.linkages:
                if link.s2==stage and link.type != 'Fridge':
                    fluxes.append(link.flux)
            stageFlux.append(fluxes)
        return stageFlux


    def showFridge(self):
        l = 20.9 #Helium heat of evaporization [J/g]

        print('Fridge:')
        for link in self.linkages:
            if link.type == 'Fridge':
                print('  Link: %10s\t'%link.name,'Avg MT Load: ', uprint(link.cycle_flux, fmt='%6.3f'),                     '')
                print('  Hold Time: %2.2f h\t %2.2f h\t %2.2f h\t'%(link.hold_time1/3600,link.hold_time2/3600,link.hold_time3/3600), )
                print('  Cycle Efficiency: %2.3f \t'%(1 - link.recycle_time/link.hold_time))
                print('  Ext Load: %s\t'%uprint(link.s2.load, fmt='%4.2f'),) #'Parasitic Load: %s\t'%uprint(link.para_load, fmt='%4.2f') )
                print('  Flowrate when cycle: %2.3f g/s'%(link.heaterE/(2*3600)/l))


    ### methods
    def calLoads(self):
        for stage in self.stages:
            stage.load = 0
        for link in self.linkages:
            flux = link.calFlux()

    # find equlibrium temp of the floating stages
    def findEq(self, gain=1, eff=[1,1], gas='He', tankP=0, epsT=1e-4 ):
        dT_max = 10
        step=0

        while dT_max>epsT:
            step+=1
            self.calLoads()

            # main tank loading with added heat
            p = self.stages[3].load + tankP

            dT_max=0
            for j in [1,2]:
                stage = self.stages[j]
                prv_stage =self.stages[j+1]
                cooling_pow = (stage.T-prv_stage.T)*cp(gas)*power2mdot(p,gas)*eff[j-1]
                dT = (stage.load-cooling_pow)*gain
                stage.T = stage.T+dT

                if np.abs(dT)> dT_max: dT_max=np.abs(dT)

            T_stages = [self.stages[1].T,self.stages[2].T,]

        print("VCS1 Temp: %1.2f"%T_stages[0])
        print("VCS2 Temp: %1.2f"%T_stages[1])
        print("In %1d steps\n"%step)

        return T_stages, step


    # cooldown process
    # def cooldown(self, LN_fillrate=10, LHE_fillrate=10, LN_):


    # cooling
    def heatEq(self, tEnd, dt, dp=0, eff=1, gas='He', wt_overwrite=0):
        t  = np.arange(0,tEnd,dt)
        T_stages = np.zeros((len(t),2))

        for i,ts in enumerate(t):
            T_stages[i] = [self.stages[1].T,self.stages[2].T,]
            self.calLoads()

            p = self.stages[3].load + dp # main tank loading in Watt
            
            # Liquid Helium 
            if gas=='He':
                mdot = p/84.5*0.004
                cp = 5193

            # Liquid Nitrogen
            elif gas=='N':
                mdot = p/2.7928e3*0.01401
                cp = 1041
            
            for j in [1,2]:
                stage =self.stages[j]
                prv_stage =self.stages[j+1]
                cooling_pow = (stage.T-prv_stage.T)*eff*cp*mdot
                if wt_overwrite:
                    dT = (stage.load-cooling_pow)*dt/(wt_overwrite *AlCp(stage.T))
                else:
                    dT = (stage.load-cooling_pow)*dt/(stage.Wt *AlCp(stage.T))
                stage.T = stage.T+dT

        return t,T_stages


    def pumpImpact(self, tEnd, dt, dp=0, dp_tStr=0,dp_t=0):
        '''
        Function to study the impact of the heat dump to the MT when cycling the pump
        Apply dp power to MT starting from dp_tStr for dp_t 

        '''

        t  = np.arange(0,tEnd,dt)
        T_stages = np.zeros((len(t),2))

        for i,ts in enumerate(t):
            T_stages[i] = [self.stages[1].T,self.stages[2].T,]
            self.calLoads()

            p = self.stages[3].load # main tank loading

            # figure out the total(thermal and fridge) loading at MT
            if ts<dp_tStr:
                mdot = p/85*0.004
            elif ts<dp_tStr+dp_t:
                mdot = (p+dp)/85*0.004
            else:
                mdot = p/85*0.004

            for j in [1,2]:
                stage =self.stages[j]
                prv_stage =self.stages[j+1]
                dT = (stage.load-(stage.T-prv_stage.T)*5.2e3*mdot)*dt/(stage.Wt *AlCp(stage.T))
                stage.T = stage.T+dT


        return t,T_stages


class Transient(object):
    ### setting up the model
    def __init__(self, name, system):
        self.name = name
        self.system = system

 


def ringArea(OD, thickness):
	'''return the area of a ring'''
	return np.pi * ((OD/2.)**2 - (OD/2.-thickness)**2)



def AlCp(T):
    # Return Al 6061 specific heat from 4-300K in J/(kg*K)
    coeff = np.vstack([46.6467,-314.292,866.662,-1298.3,1162.27,-637.795,210.351,-38.3094,2.96344])
    logT  = np.log10(T)
    logCv = np.power(logT, np.vstack(np.arange(0,9,1))) *coeff

    return 10**logCv.sum(axis=0)[0]

# need to change the following He functions into classes


def cp(gas='He'):
    # return specific heat of gas in J/(Kg*K)
    if gas == 'He':
        return 5193
    elif gas == 'N':
        return 1041


def power2mdot(p, gas='He'):
    if gas == 'He':
        L = 84.5/0.004
    elif gas == 'N':
        L = 2.7928e3/0.01401
    return p/L


def mdot2power(mdot, gas='He'):
    if gas == 'He':
        L = 84.5/0.004
    elif gas == 'N':
        L = 2.7928e3/0.01401
    return mdot*L

    
def mdot2SLPM(mdot, gas='He'):
    '''
    Converting from kg/s to standard liters per minute (SLPM)
    '''
    if gas == 'He':
        DensATStp = 4.003/22.41*1e-3 #kg/L
    elif gas == 'N':
        DensATStp = 28.02/22.41*1e-3 #kg/L

    SLPM = 60 * ( mdot / DensATStp )
    return SLPM

if __name__ == '__main__':
    print(1)
