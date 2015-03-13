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
	(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell) = (1.5, 4.3, 10., 100., args.VVTemp)
	#(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell) = (1.5, 4.3, 10., 100., 280.)
	mdot = 0.030
	
	#Cooling efficiencies
	e_23 = 1    #cooling efficiency between MT and VCS1
	e_34 = 0.9  #cooling efficiency between VCS1 and VCS2
  	
	sftPumped = True
	#sftEmpty = False
	sftEmpty = args.sftEmpty
	
	if args.ULDB:
		insNum = 1.0
		config = 'ULDB'
		numLiters = 120.0
	elif args.ULDB2:
		insNum = 1.0
		config = 'ULDB2'
		numLiters = 120.0		
	elif args.theo2:
		config = 'theo2'
		insNum = 6
		numLiters = 1000.0
	elif args.theo1:
		config = 'theo1'
		insNum = 6
		numLiters = 1000.0
	elif args.theo_alt1:
		config = 'theo_alt1'
		insNum = 6
		numLiters = 1000.0
	else:
		config = 'theo'
		insNum = 6
		numLiters = 1000.0
	
	#place holders for filter loads
	#window_MT =  insNum*0.05 #50 mW estimate
	#window_VCS1 = insNum*0.7 # 0.7W estimate from Theo paper #/6
	
	capLoad = 0.008 #~50mW / 6 for capillary box
				
	# filter model
	# TODO: add options for different loads at the aperture
	# (cf. radmodel.main())
	if args.mylarWindow:
	        radmodel_params = models['ar_mylarwindow_nonylon']
        elif args.shaderWindow:
                radmodel_params = models['ar_nonylon_windowshader']
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
	#gain = 0.025
	gain = 0.05
	#gain = 0.1
	    
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
		if args.windowsOpen:	
			inband, window_MT, window_VCS1, window_VCS2 = \
				filter_load(M, T_SFT, T_MT, T_VCS1, T_VCS2, 273,
				insNum, **radmodel_params)
		else:
			inband, window_MT, window_VCS1, window_VCS2 = np.zeros(4)
			insNum = 0.0
			

		if args.keller:
			
			Rad_SFTtoMT, RadSFTtoVCS1, Rad_MT, Rad_VCS1, Rad_VCS2 = \
				mli_rad_keller(T_SFT, T_MT, T_VCS1, T_VCS2, T_Shell, 
					p_ins=1e-4, e_Al=0.15, alpha=0.15, beta=4.0e-3, config=config, insNum = insNum)
			Rad_SFT = Rad_SFTtoMT+RadSFTtoVCS1
					
		else:			
			mli_load_VCS1, mli_load_VCS2 = mli_cond(T_VCS1, T_VCS2, T_Shell, config = config, insNum = insNum)
		
			Rad_SFTtoMT, RadSFTtoVCS1, Rad_MT, Rad_VCS1, Rad_VCS2 = rad_load(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell,
				e_Al=0.15, alpha=0.15, beta=4.0e-3, config = config, insNum = insNum)
			Rad_SFT = Rad_SFTtoMT+RadSFTtoVCS1
			
			Rad_VCS1 += mli_load_VCS1  #is this appropriate?  some of this goes to cooling, should all of it?
			Rad_VCS2 += mli_load_VCS2
		
		#print('TVCS2: %s, VCS2 window power: %s' % (T_VCS2, window_VCS2))
		#print('TVCS1: %s, VCS1 window power: %s' % (T_VCS1, window_VCS1))
		#print('MT window power: %s' % window_MT)
 
		##CONDUCTION through flexures and stainless tubes##
		(tubeCondLoad1, tubeCondLoad2, tubeCondLoad4In, tubeCondLoad4Out, 
		    flexCondLoad1, flexCondLoad2in, flexCondLoad2out, flexCondLoad3In, 
		    flexCondLoad3Out, flexCondLoad4In, flexCondLoad4Out) = cond_loads(T_SFT,T_MT,T_VCS1,T_VCS2,T_Shell,
				sftPumped,sftEmpty,insNum, config = config, flexFactor = args.flexFactor)
		cfact = 1.2		
		tubeCondLoad_SFT = cfact*tubeCondLoad1
		tubeCondLoad_MT = cfact*tubeCondLoad2
		tubeCondLoad_VCS2 = cfact*(tubeCondLoad4In + tubeCondLoad4Out)
		
		flexCondLoad_SFT = cfact*flexCondLoad1
		flexCondLoad_MT = cfact*(flexCondLoad2in + flexCondLoad2out)
		flexCondLoad_VCS1 = cfact*(flexCondLoad3In + flexCondLoad3Out)
		flexCondLoad_VCS2 = cfact*(flexCondLoad4In + flexCondLoad4Out)
		#gas cooling power
		gasCoolingVCS1 = e_23*mdot*CpInt(T_MT, T_VCS1, T_He, Cp_He)
		gasCoolingVCS2 = e_34*mdot*CpInt(T_VCS1, T_VCS2, T_He, Cp_He)
		l = 21. #Helium heat of evaporization [J/g]
	   	
		MTexcess = args.mtExcess #excess load in watts?	
		VCS2excess = args.VCS2Excess #excess load in watts?	
		MT = capLoad + Rad_MT + window_MT  \
				- Rad_SFTtoMT
		MT += (tubeCondLoad_MT + flexCondLoad_MT - tubeCondLoad_SFT)
		MT += MTexcess
		
		MTLoad = capLoad + Rad_MT + window_MT \
				+ flexCondLoad2in + tubeCondLoad_MT \
				+ MTexcess
				
		SFTLoad = Rad_SFT
		SFTLoad += (tubeCondLoad_SFT + flexCondLoad_SFT)
		
		#cryocooler parameters:
		T, lift = np.loadtxt('cryotel_GT_23C.txt', unpack = True, delimiter = ',')
		p = np.polyfit(T, lift, 1) #fitting a line for now since the cooling curve is close
		lowT = np.array([20, 30, 40])
		lowlift = np.array([0.124, 0.223, 0.3])
		p_low = np.polyfit(lowT, lowlift, 1) #linear fit to lower temp data from AKARI
			
		icsCryocooler = args.icsCoolers*np.max([np.polyval(p, T_VCS1), np.polyval(p_low, T_VCS1)])
				
		VCS1 = Rad_VCS1 + window_VCS1 \
				-Rad_MT - RadSFTtoVCS1 - gasCoolingVCS1 - icsCryocooler \
				-MTexcess
		
		VCS1_load = Rad_VCS1 + window_VCS1
		
		VCS1 +=  flexCondLoad_VCS1 
		VCS1_load += cfact*flexCondLoad3In
		
		ocsCryocooler = args.ocsCoolers*np.max([np.polyval(p, T_VCS2), np.polyval(p_low, T_VCS2)])
		
		VCS2 = Rad_VCS2  + window_VCS2 \
				-Rad_VCS1 - gasCoolingVCS2 - ocsCryocooler
		VCS2_load = Rad_VCS2  + window_VCS2 
		
		VCS2 +=  flexCondLoad_VCS2 + tubeCondLoad_VCS2 - tubeCondLoad_MT
		VCS2 += VCS2excess
		
		VCS2_load += cfact*flexCondLoad4In + cfact*tubeCondLoad4In
		VCS2_load += VCS2excess
		
		mdot = MT / l
		mdot2 = MTLoad / l
		
		#print VCS1, VCS2, mdot
		#Check if loads ~ zero
		if (abs(VCS1) < eps and abs(VCS2) < eps):
		
			print('ICS cryocooler power: %s' % icsCryocooler)
			print('ICS gas cooling power: %s' % gasCoolingVCS1)
			
			print('OCS cryocooler power: %s' % ocsCryocooler)
			print('OCS gas cooling power: %s' % gasCoolingVCS2)
						
			print('mdot (g/s): %s, Holdtime (days): %s' % (mdot, holdtime(mdot, numLiters = numLiters)))
			print('SLPM (from MT/l): %s' % (mdot2SLPM(mdot)))
			print('SLPM (from MTLoad/l): %s' % (mdot2SLPM(mdot2)))
			#print summary
			print(' Stage |Temperature |')
			print(' OCS   | %1.2f K |' % T_VCS2)
			print(' ICS   | %1.2f K |' % T_VCS1)
			print('--------')
			print('Loads')
			print('          |   SFT      |    MT      |   ICS      |   OCS      |')
			print('Aperture  | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' % (0.0, window_MT, window_VCS1, window_VCS2))
			print('Radiative | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' % (Rad_SFT, Rad_MT, Rad_VCS1, Rad_VCS2))
			#print('MLI       | %1.2e W | %1.2e W | %1.2e W |' % (0.0, mli_load_VCS1, mli_load_VCS2))
			print('Flexures  | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' \
				% (cfact*flexCondLoad_SFT, cfact*flexCondLoad2in, cfact*flexCondLoad3In, cfact*flexCondLoad4In))
			print('Plumbing  | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' \
				% (tubeCondLoad_SFT, tubeCondLoad_MT, 0, cfact*tubeCondLoad4In))
			print('--------')
			print('Total     | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' % (SFTLoad, MTLoad, VCS1_load, VCS2_load))
			print('In-band detector loading: %s' % uprint(inband))
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
	parser.add_argument('-theo2', dest='theo2', action='store_true', help='Run theo model with large flexure intercept only to VCS2')
	parser.add_argument('-theo_alt1', dest='theo_alt1', action='store_true', help='Run theo model with as built dims, but connected only to VCS1 intercept')
	parser.add_argument('-theo1', dest='theo1', action='store_true', help='Run theo model with large flexure intercept only to VCS1')
	parser.add_argument('-flexFact', dest = 'flexFactor', action = 'store', type=float, default=1.0, help='Reduction factor in flexure conduction')
	parser.add_argument('-ocsCoolers', dest = 'ocsCoolers', action = 'store', type = int, default = 0.0, help='Number of OCS coolers')
	parser.add_argument('-icsCoolers', dest = 'icsCoolers', action = 'store', type = int, default = 0.0, help='Number of ICS coolers')
	parser.add_argument('-mylarWindow', dest = 'mylarWindow', action = 'store_true', help='Use a 10-um Mylar window instead of the default 1/8" PE')
	parser.add_argument('-shaderWindow', dest = 'shaderWindow', action = 'store_true', help='Use a PE window with a 300K shader.')
	parser.add_argument('-keller', dest = 'keller', action = 'store_true', help='Use the keller MLI model')
	parser.add_argument('-sftEmpty', dest = 'sftEmpty', action = 'store_true', help='Model with the SFT empty')
	parser.add_argument('-noInserts', dest = 'windowsOpen', action = 'store_false', help='Run with no inserts installed (no filters)')
	parser.add_argument('-mtExcess', dest = 'mtExcess', action = 'store', type = float, default= 0.0, help='Excess load on MT, in Watts')
	parser.add_argument('-VCS2Excess', dest = 'VCS2Excess', action = 'store', type = float, default= 0.0, help='Excess load on VCS2, in Watts')
	parser.add_argument('-VVTemp', dest = 'VVTemp', action = 'store', type = float, default= 248.0, help='Vacuum vessel wall temperature')

	args = parser.parse_args()
	
	T_VCS1 , T_VCS2, mdot = find_equilibrium(args)
	
