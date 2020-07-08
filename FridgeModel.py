#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as sp

from scipy.integrate import simps


from radmodel import uprint


#constants
#Stefan-Boltzmann constant
SIGMA = 5.6704E-12   #[J/s*cm^2*K^4]


class ThermalMaterial:
    '''
    A class for materials used in the thermal modeling
    '''
    def __init__(self, name, specificHeat=np.inf, conductivity=np.inf, latentHeat=np.inf ):
        self.name = name
        self.specificHeat = specificHeat
        self.conductivity = conductivity
        self.latentHeat = latentHeat 
        
    def getFlux(self,T1,T2,AOverL):
        
        Ts = self.conductivity[:,0]
        ks = self.conductivity[:,1]
        
        dT = np.min(Ts[1:] - Ts[:-1])
        T  = np.arange(T1, T2, dT)
        
        return AOverL * simps(np.interp(T, Ts, ks), T)

SShigh = np.loadtxt('./thermalProp/ss304_cv.txt',)
SST    = np.arange(.2,3.9,0.1)
SSlow  = np.stack((SST, 0.0556*SST**1.15)).T



# Common materials
SS = ThermalMaterial(name='SS304', conductivity=np.concatenate((SSlow,SShigh)))
CF = ThermalMaterial(name='CF', conductivity=np.loadtxt('./thermalProp/CF_cv.txt',))
G10 = ThermalMaterial(name='G10', conductivity=np.loadtxt('./thermalProp/g10_cv_combined.txt',))
    
        
class ThermalLink(object):
    def __init__(self, name, s1, s2,):
        self.name = name
        self.s1   = s1
        self.s2   = s2

class Radi(ThermalLink):
    def __init__(self, area=0, **kwargs):
        super().__init__(**kwargs)
        self.type = 'Radiation'
        self.area = area 
        
    def calFlux(self):
        self.flux = self.area * SIGMA * (self.s1.T**4 - self.s2.T**4)
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
        
class Fridge(ThermalLink):
    '''
    This is a special link class that represents the Fridge.
    Useful quantities will be stored and calculated using this class.
    Noted that the flux(or load) calculated using calFlux method would not be
    added to the stages as loads on stages are specified to external loads.
    '''

    def __init__(self, stp = 8, heaterE=5000, no_load_time=48*3600, cycle_time=2*3600,\
        FPU_heat_cap=2.74, **kwargs):
        '''
        stp [L]: Helium 3 volume in stp liters
        heaterE [J]: Energy from the fridge heaters in one cycle
        no_load_time [s]: Hold time of the fridge still when there is no external loads
        cycle_time [s]: Time to cycle the fridge, i.e. time that still is not at operating temp
        FPU_heat_cap [J]: Energy needed to cool the cold element at the begining 
        of each cycle(from 4k to 0.3k), 
        2.74 = 2.5(two and half module)* 0.58(volume) * 2.7(Al density) * 0.7(Al Enthalpy[J/kg] at 4K) 
        '''
        
        super().__init__(**kwargs)
        self.type = 'Fridge'
        # total cooling power at the still in J,
        # He3 latent heat is around 25 J/mol at 0.3 K
        self.coldE = stp/22.4*25 
        self.heaterE = heaterE   
        self.no_load_time = no_load_time
        self.para_load = self.coldE/no_load_time  # still parasitic load in W
        self.FPU_heat_cap = FPU_heat_cap

        self.cycle_time = cycle_time
        
    def getStillLoad(self):
        self.still_load = self.s2.load + self.para_load
        self.hold_time = (self.coldE-self.FPU_heat_cap)/ self.still_load

    def calFlux(self):
        self.getStillLoad()
        self.cycle_flux = self.heaterE/self.hold_time

        return 0
        
class Stages(object):
    def __init__(self, name, T0, Type=None):
        self.name = name
        self.T    = float(T0)
        self.Type = Type
        self.load = 0.
        
        
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
                print('  %s\t\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), \
                    '| %s\t'%link.type, '| %s, A/L: %3.2e m'%(link.material.name, link.AOverL))
                
            elif link.type == 'Radiation':
                print('  %s\t\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), \
                     '| %s\t'%link.type, '| area: %3.2e cm^2'%link.area)
            else:
                print('  %s\t\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), '|', link.type)
            
    def showStages(self,):
        print('Stages:')
        for stage in self.stages:
            print('  ', stage.name,', Temp: %1.2fK'%stage.T)
    
    def showSystem(self):
        print('System:', self.name)
        self.showStages()
        self.showLinks()
        
    def showLoads(self):
        print('Fluxs on links:')
        for link in self.linkages:
            if link.type != 'Fridge':
                print('  Link: %10s\t'%link.name,'| %s to %s\t'%(link.s1.name, link.s2.name), '| Flux: ', uprint(link.flux, fmt='%6.3f') )
        print('Static Loads on stages:')
        for stage in self.stages:
            print('  Stage: %s\t'%stage.name,' Load: ', uprint(stage.load, fmt='%6.3f') )
         
        
    def showFridge(self):
        l = 20.9 #Helium heat of evaporization [J/g]
        
        print('Fridge:')
        for link in self.linkages:
            if link.type == 'Fridge':
                print('  Link: %10s\t'%link.name,'Avg MT Load: ', uprint(link.cycle_flux, fmt='%6.3f'),                     '')
                print('  Hold Time: %2.2f h\t'%(link.hold_time/3600), 'Cycle Efficiency: %2.3f \t'%(1 - link.cycle_time/link.hold_time))
                print('  Ext Load: %s\t'%uprint(link.s2.load, fmt='%4.2f'), 'Parasitic Load: %s\t'%uprint(link.para_load, fmt='%4.2f') )
                print('  Flowrate when cycle: %2.3f g/s'%(link.heaterE/(2*3600)/l))
                
                
    ### methods
    def calLoads(self):
        for stage in self.stages:
            stage.load = 0
        for link in self.linkages:
            flux = link.calFlux()


if __name__ == '__main__':
    print(1)

        

