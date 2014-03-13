#!/usr/bin/env python
# encoding: utf-8
"""
find_thermal_eq.py

Created by Zigmund Kermish on 2014-01-20.  Heavily copy/pasted from Jon Gudmundsson's Matlab code, which was based
on Bill Jones' IDL code
"""

import sys
import os
from radiative_loads import *
from conductive_loads import *
from gas_props import *
import argparse

from radmodel import *

def find_equilibrium(args):
	#Calculating the starting net heat load on VCS1 and VCS2
	VCS1 = 1
	VCS2 = 2
	#setting initial temperatures and flows -20C = 253.15
	(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell) = (1.5, 4.2, 10., 100., 300.)
	mdot = 0.030
	
	#Cooling efficiencies
	e_23 = 1    #cooling efficiency between MT and VCS1
	e_34 = 0.9  #cooling efficiency between VCS1 and VCS2
  	
	sftPumped = True
	sftEmpty = False
	
	if args.ULDB:
		insNum = 1.0
		config = 'ULDB'
		numLiters = 120.0
	elif args.ULDB2:
		insNum = 1.0
		config = 'ULDB2'
		numLiters = 120.0		
	else:
		config = 'theo'
		insNum = 6
		numLiters = 1000.0
	
	#place holders for filter loads
	#window_MT =  insNum*0.05 #50 mW estimate
	#window_VCS1 = insNum*0.7 # 0.7W estimate from Theo paper #/6
	
	capLoad = 0.008 #~50mW / 6 for capillary box
	
	window_MT =  insNum*0.004 #4 mW estimate from current radmodel code with nylon as upper bound
	window_VCS1 = insNum*0.030 # 2x current radmodel code estimate 		
	window_VCS2 =  insNum*1.2 # current radmodel code estimate
	
	#window_VCS1 = insNum*0.13 # 2x current radmodel code estimate 		
	#window_VCS2 =  insNum*7 # current radmodel code estimate
	#window_MT =  insNum*0.01 #10 mW estimate from current radmodel code 
	#window_VCS1 = insNum*0.030 # 0.030W estimate from current radmodel code
        
	#window_MT =  insNum*0.08 #80 mW estimate from current radmodel code 
	#window_VCS1 = insNum*1.0 # arbitrary nylon hot potato #
		
	
        # filter model
        # TODO: add options for different loads at the aperture
        # (cf. radmodel.main())
        if args.mylarWindow:
                radmodel_params = models['ar_mylarwindow_nonylon']
        else:
                radmodel_params = models['ar_nonylon']
        M = RadiativeModel()
        
	#Counter and maximum number of iterations
	n = 1
	maxIter = 500
	
	#tolerance
	eps = 0.02
	sfteps = 0.0005
	DeltaT = 0.02
	# gain = 0.025
	gain = 0.05
        
	while n <=maxIter:
		if (abs(VCS1) > abs(VCS2)):
			if (VCS1 > 0):
				T_VCS1 = T_VCS1 + DeltaT*abs(VCS1)/gain
			else:
				T_VCS1 = T_VCS1 - DeltaT*abs(VCS1)/gain
		else:
			if (VCS2 > 0):
				T_VCS2 = T_VCS2 + DeltaT*abs(VCS2)/gain
			else:
				T_VCS2 = T_VCS2 - DeltaT*abs(VCS2)/gain	
				   
		#update loads
		
		#Radiative loads and MLI conductivity estimates
		#TODO-update with better MLI model
		mli_load_VCS1, mli_load_VCS2 = mli_cond(T_VCS1, T_VCS2, T_Shell, config = config)
		
		Rad_SFTtoMT, RadSFTtoVCS1, Rad_MT, Rad_VCS1, Rad_VCS2 = rad_load(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell,
			e_Al=0.15, alpha=0.15, beta=4.0e-3, config = config)
		Rad_SFT = Rad_SFTtoMT+RadSFTtoVCS1
		
                window_MT, window_VCS1, window_VCS2 = \
                        filter_load(M, T_SFT, T_MT, T_VCS1, T_VCS2, 273,
                                    insNum, **radmodel_params)
                print('VCS2 window power: %s' % window_VCS2)
                print('VCS1 window power: %s' % window_VCS1)
                print('MT window power: %s' % window_MT)
                
		##CONDUCTION through flexures and stainless tubes##
		(tubeCondLoad1, tubeCondLoad2, tubeCondLoad4In, tubeCondLoad4Out, 
		    flexCondLoad1, flexCondLoad2, flexCondLoad3In, 
		    flexCondLoad3Out, flexCondLoad4In, flexCondLoad4Out) = cond_loads(T_SFT,T_MT,T_VCS1,T_VCS2,T_Shell,
				sftPumped,sftEmpty,insNum, config = config, flexFactor = args.flexFactor)
		cfact = 1.2		
		tubeCondLoad_SFT = cfact*tubeCondLoad1
		tubeCondLoad_MT = cfact*tubeCondLoad2
		tubeCondLoad_VCS2 = cfact*(tubeCondLoad4In + tubeCondLoad4Out)
		
		flexCondLoad_SFT = cfact*flexCondLoad1
		flexCondLoad_MT = cfact*flexCondLoad2
		flexCondLoad_VCS1 = cfact*(flexCondLoad3In + flexCondLoad3Out)
		flexCondLoad_VCS2 = cfact*(flexCondLoad4In + flexCondLoad4Out)
		#gas cooling power
		gasCoolingVCS1 = e_23*mdot*CpInt(T_MT, T_VCS1, T_He, Cp_He)
		gasCoolingVCS2 = e_34*mdot*CpInt(T_VCS1, T_VCS2, T_He, Cp_He)
		l = 21. #Helium heat of evaporization [J/g]
	   		
		MTLoad = capLoad + Rad_MT + window_MT  \
				- Rad_SFTtoMT
		MTLoad += (tubeCondLoad_MT + flexCondLoad_MT - tubeCondLoad_SFT)
		
		SFTLoad = Rad_SFT
		SFTLoad += (tubeCondLoad_SFT + flexCondLoad_SFT)
		
		#cryocooler parameters:
		T, lift = np.loadtxt('cryotel_GT_23C.txt', unpack = True, delimiter = ',')
		p = np.polyfit(T, lift, 1) #fitting a line for now since the cooling curve is close
		lowT = np.array([20, 30, 40])
		lowlift = np.array([0.124, 0.223, 0.3])
		p_low = np.polyfit(lowT, lowlift, 1) #linear fit to lower temp data from AKARI
			
		icsCryocooler = args.icsCoolers*np.max([np.polyval(p, T_VCS1), np.polyval(p_low, T_VCS1)])
				
		VCS1 = Rad_VCS1 + mli_load_VCS1 + window_VCS1 \
				-Rad_MT - RadSFTtoVCS1 - gasCoolingVCS1 - icsCryocooler 
		
		VCS1_load = Rad_VCS1 + mli_load_VCS1 + window_VCS1
		
		VCS1 +=  (flexCondLoad_VCS1 - flexCondLoad_MT) 
		VCS1_load += flexCondLoad_VCS1
		
		ocsCryocooler = args.ocsCoolers*np.max([np.polyval(p, T_VCS2), np.polyval(p_low, T_VCS2)])
		
		VCS2 = Rad_VCS2 + mli_load_VCS2 + window_VCS2 \
				-Rad_VCS1 - gasCoolingVCS2 - ocsCryocooler
		VCS2_load = Rad_VCS2 + mli_load_VCS2 + window_VCS2 
		
		VCS2 +=  flexCondLoad_VCS2 + tubeCondLoad_VCS2 - tubeCondLoad_MT
		VCS2_load += flexCondLoad_VCS2 + tubeCondLoad_VCS2
		
		mdot = MTLoad / l
		
		#print VCS1, VCS2, mdot
		#Check if loads ~ zero
		if (abs(VCS1) < eps and abs(VCS2) < eps):
		
			print('ICS cryocooler power: %s' % icsCryocooler)
			print('ICS gas cooling power: %s' % gasCoolingVCS1)
			
			print('OCS cryocooler power: %s' % ocsCryocooler)
			print('OCS gas cooling power: %s' % gasCoolingVCS2)
						
			print('mdot (g/s): %s, Holdtime (days): %s' % (mdot, holdtime(mdot, numLiters = numLiters)))
			
			#print summary
			print(' Stage |Temperature |')
			print(' OCS   | %1.2f K |' % T_VCS2)
			print(' ICS   | %1.2f K |' % T_VCS1)
			
			print('Loads')
			print('          |    MT   |   ICS   |   OCS   |')
			print('Aperture  | %1.2e W | %1.2e W | %1.2e W |' % (window_MT, window_VCS1, window_VCS2))
			print('Radiative | %1.2e W | %1.2e W | %1.2e W |' % (Rad_MT, Rad_VCS1, Rad_VCS2))
			print('MLI       | %1.2e W | %1.2e W | %1.2e W |' % (0.0, mli_load_VCS1, mli_load_VCS2))
			print('Flexures  | %1.2e W | %1.2e W | %1.2e W |' % (flexCondLoad_MT, flexCondLoad_VCS1, flexCondLoad_VCS2))
			print('Plumbing  | %1.2e W | %1.2e W | %1.2e W |' % (tubeCondLoad_MT, 0, tubeCondLoad_VCS2))
			print('Total     | %1.2e W | %1.2e W | %1.2e W |' % (MTLoad, VCS1_load, VCS2_load))
			
			return T_VCS1 , T_VCS2, mdot
		# Cutting back on our precision
		if ( n == np.floor(maxIter/2) ):
			DeltaT = DeltaT/2
			eps = eps*1.5
			print(n)
		if ( n == np.floor(0.75*maxIter) ):
			DeltaT = DeltaT/2
			eps = eps*1.5
			print(n)
		n += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Thermal model')
	parser.add_argument('-ULDB', dest='ULDB', action='store_true', help='Run ULDB model instead of Theo?')
	parser.add_argument('-ULDB2', dest='ULDB2', action='store_true', help='Run ULDB *2* model instead of Theo?')
	parser.add_argument('-flexFact', dest = 'flexFactor', action = 'store', type=float, default=1.0, help='Reduction factor in flexure conduction')
	parser.add_argument('-ocsCoolers', dest = 'ocsCoolers', action = 'store', type = int, default = 1.0, help='Number of OCS coolers')
	parser.add_argument('-icsCoolers', dest = 'icsCoolers', action = 'store', type = int, default = 0.0, help='Number of ICS coolers')
        parser.add_argument('-mylarWindow', dest = 'mylarWindow', action = 'store_true', help='Use a 10-um Mylar window instead of the default 1/8" PE')

	args = parser.parse_args()
	
	T_VCS1 , T_VCS2, mdot = find_equilibrium(args)
	
