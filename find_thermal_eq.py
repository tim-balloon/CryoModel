#!/usr/bin/env python
# encoding: utf-8
"""
find_thermal_eq.py

Created by Zigmund Kermish on 2014-01-20.
Heavily copy/pasted from Jon Gudmundsson's Matlab code,
which was based on Bill Jones' IDL code
"""

import sys
import os

from time import time

_this_dir, _this_filename = os.path.split(__file__)

from radiative_loads import *
from conductive_loads import *
from gas_props import *
import argparse

from radmodel_new2 import *
# from radmodel import *
from wiring import wiring_load

def find_equilibrium(args):

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
	elif args.lloro:
		config = 'lloro'
		insNum = 6
		numLiters = 1000.0
	elif args.theo_alt1:
		config = 'theo_alt1'
		insNum = 6
		numLiters = 1000.0
	elif args.TNG:
		config = 'TNG'
		insNum = 1
		numLiters = 250.0
	elif args.TIM:
		config = 'TIM'
		insNum = 1
		numLiters = 250.0
	else:
		config = 'theo'
		insNum = 6
		numLiters = 1100.0


	#Calculating the starting net heat load on VCS1 and VCS2
	VCS1 = 1.
	VCS2 = 1.1

	#setting initial temperatures and flows -20C = 253.15
	if config == 'TNG' or config=='TIM':
		(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell) = (1.75, 4.2, 65, 170, args.VVTemp)
	else:
		(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell) = (1.5, 4.3, 40., 110., args.VVTemp)

	mdot = 0.030

	#Cooling efficiencies
	if config == 'TNG'  or config=='TIM':
		effi_23 = 0.8 # 1    #cooling efficiency between MT and VCS1
		effi_34 = 0.8 # 0.9  #cooling efficiency between VCS1 and VCS2

	else:
		effi_23 = 0.9 # 1    #cooling efficiency between MT and VCS1
		effi_34 = 0.9 # 0.9  #cooling efficiency between VCS1 and VCS2

	sftPumped = True

	sftEmpty = args.sftEmpty


	#place holders for filter loads
	#window_MT =  insNum*0.05 #50 mW estimate
	#window_VCS1 = insNum*0.7 # 0.7W estimate from Theo paper #/6
	#capLoad = 0.008 #~50mW / 6 for capillary box



	if config != 'TNG' and config!='TIM':

		if args.windowsOpen:
			capLoad = 0.123 #~2SLPM
		else:
			capLoad = 0
		# filter model
		# TODO: add options for different loads at the aperture
		# (cf. radmodel.main())
		if args.mylarWindow:
			radmodel_params = models['ar_mylarwindow_nonylon']
		elif args.shaderWindow:
			radmodel_params = models['ar_nonylon_windowshader2']
		else:
				radmodel_params = models['ar_nonylon']
		if args.ground:
			opts = {}
			#opts['atmfile'] = 'am_01km.dat'
			opts['tsky'] = 300.0
			opts['atmos']= False
		else:
			opts = {}
		M = RadiativeModel(**opts)

	T0 = time()

	print('Configuration: %s'%config)
	print('T_Shell: %.1f K'%T_Shell)

	#Counter and maximum number of iterations
	n = 1
	maxIter = 500

	#tolerance
	eps = 0.05 #0.02
	sfteps = 0.0005

	gain = 2

	while n <= maxIter:

		if (abs(VCS1) > abs(VCS2)):
			T_VCS1 = T_VCS1 + VCS1*gain
		else:
			T_VCS2 = T_VCS2 + VCS2*gain

		# print(VCS1, VCS2,)
		# print(T_VCS1, T_VCS2)
		#update loads

		#Radiative loads and MLI conductivity estimates

		#toy model for TNG filter loads (from excel model)
		if config == 'TNG'  or config=='TIM':
			window_MT, window_VCS1, window_VCS2 = toy_filter_load(T_SFT, T_MT, \
				T_VCS1, T_VCS2, T_Shell, config=config, \
				insNum=insNum)
			inband = 0

			#cap loading from spider
			capLoad = 1e-3  #~2SLPM

		else:
			if args.windowsOpen:
				inband, window_MT, window_VCS1, window_VCS2 = \
					filter_load(M, T_SFT, T_MT, T_VCS1, T_VCS2, T_Shell,
					insNum, **radmodel_params)
			else:
				inband, window_MT, window_VCS1, window_VCS2 = np.zeros(4)
				insNum = 0.0




		if args.keller:

			Rad_MTtoSFT, Rad_VCS1toSFT, Rad_MT, Rad_VCS1, Rad_VCS2 = \
				mli_rad_keller(
					T_SFT, T_MT, T_VCS1, T_VCS2, T_Shell,
					p_ins1=args.pins1, p_ins2=args.pins2,
					e_Al=0.15, alpha=0.15, beta=4.0e-3,
					config=config, insNum=insNum)
			Rad_SFT = Rad_MTtoSFT + Rad_VCS1toSFT
			#Rad_VCS2 *= 1.5

		else:
			mli_load_VCS1, mli_load_VCS2 = mli_cond(T_VCS1, T_VCS2, T_Shell, config = config, insNum = insNum)

			Rad_MTtoSFT, Rad_VCS1toSFT, Rad_MT, Rad_VCS1, Rad_VCS2 = rad_load(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell,
				e_Al=0.15, alpha=0.15, beta=4.0e-3, config = config, insNum = insNum)
			Rad_SFT = Rad_MTtoSFT + Rad_VCS1toSFT

			Rad_VCS1 += mli_load_VCS1  #is this appropriate?  some of this goes to cooling, should all of it?
			Rad_VCS2 += mli_load_VCS2



		#print('TVCS2: %s, VCS2 window power: %s' % (T_VCS2, window_VCS2))
		#print('TVCS1: %s, VCS1 window power: %s' % (T_VCS1, window_VCS1))
		#print('MT window power: %s' % window_MT)

		# Conduction through electrical wiring
		wdict = wiring_load(t_sft=T_SFT, t_mt=T_MT, t_vcs1=T_VCS1,
							t_vcs2=T_VCS2, t_vv=T_Shell, num_inserts=insNum, verbose=False)

		wire_VCS1_in = wdict['vv_vcs1']
		wire_VCS1_out = wdict['vcs1_mt']
		wire_MT = wdict['vcs1_mt']

		# CONDUCTION through flexures and stainless tubes
		(tubeCondLoad1, tubeCondLoad2In, tubeCondLoad2Out, tubeCondLoad3In,
			tubeCondLoad3Out, tubeCondLoad4In, tubeCondLoad4Out,
			flexCondLoad1, flexCondLoad2In, flexCondLoad2out, flexCondLoad3In,
			flexCondLoad3Out, flexCondLoad4In, flexCondLoad4Out) = cond_loads(T_SFT,T_MT,T_VCS1,T_VCS2,T_Shell,
				sftPumped,sftEmpty,insNum, config = config, flexFactor = args.flexFactor)

		cfact = 1.
		tubeCondLoad_SFT = cfact*tubeCondLoad1
		tubeCondLoad_MT = cfact*(tubeCondLoad2In + tubeCondLoad2Out)
		tubeCondLoad_VCS1 = cfact*(tubeCondLoad3In + tubeCondLoad3Out)
		tubeCondLoad_VCS2 = cfact*(tubeCondLoad4In + tubeCondLoad4Out)

		flexCondLoad_SFT = cfact*flexCondLoad1
		flexCondLoad_MT = cfact*(flexCondLoad2In + flexCondLoad2out)
		flexCondLoad_VCS1 = cfact*(flexCondLoad3In + flexCondLoad3Out)
		flexCondLoad_VCS2 = cfact*(flexCondLoad4In + flexCondLoad4Out)


		#gas cooling power
		gasCoolingVCS1 = effi_23*mdot*CpInt(T_MT, T_VCS1, T_He, Cp_He)
		gasCoolingVCS2 = effi_34*mdot*CpInt(T_VCS1, T_VCS2, T_He, Cp_He)
		l = 20.9 #Helium heat of evaporization [J/g]

		MTexcess = args.mtExcess #excess load in watts. coming from VCS1

		SFT_Area, MT_Area, VCS1_Area, VCS2_Area = areas.load_areas(config=config, insNum = insNum)

		# Excessive loading per args settings
		scale=4.
		MTexcessShell = sigma*(args.mtLLVCS2perc/100.)*MT_Area*(T_Shell**scale - T_MT**scale) #direct LL from shell, not cooling power
		MTexcess2 = sigma*(args.mtLLVCS1perc/100.)*MT_Area*(T_VCS2**scale - T_MT**scale) #LL from VCS2, cools VCS2

		VCS1excess = args.VCS1Excess #excess load on vcs1 in watts
		VCS1excessShell = sigma*(args.VCS1LLperc/100.)*VCS1_Area*(T_Shell**scale-T_VCS1**scale)

		VCS2excess = args.VCS2Excess #excess load on vcs2 in watts

		n_LNA = 5
		LNA = n_LNA * 5e-3

		SFTLoad = Rad_SFT
		SFTLoad += (tubeCondLoad_SFT + flexCondLoad_SFT)

		# MT is the net	load on the main tank
		# MTLoad is the input load on the main tank
		# MT = MTLoad - SFTLoad

		MTLoad = capLoad + Rad_MT + window_MT \
				+ cfact*flexCondLoad2In + tubeCondLoad_MT \
				+ MTexcess + MTexcess2 + MTexcessShell \
				+ wire_MT + LNA

		MT = MTLoad - SFTLoad



		#cryocooler parameters:
		if (args.icsCoolers + args.ocsCoolers) > 1e-6 : # if there is cryo cooler
			T, lift = np.loadtxt('./cryoCooler/cryotel_GT_23C.txt', unpack = True, delimiter = ',')
			p = np.polyfit(T, lift, 1) #fitting a line for now since the cooling curve is close
			lowT = np.array([20, 30, 40])
			lowlift = np.array([0.124, 0.223, 0.3])
			p_low = np.polyfit(lowT, lowlift, 1) #linear fit to lower temp data from AKARI

			icsCryocooler = args.icsCoolers*np.max([np.polyval(p, T_VCS1), np.polyval(p_low, T_VCS1)])
			ocsCryocooler = args.ocsCoolers*np.max([np.polyval(p, T_VCS2), np.polyval(p_low, T_VCS2)])
		else:
			icsCryocooler = 0
			ocsCryocooler = 0

		VCS1 = Rad_VCS1 + window_VCS1 \
				-Rad_MT - Rad_VCS1toSFT - gasCoolingVCS1 - icsCryocooler \
				-MTexcess

		VCS1 += flexCondLoad_VCS1 + tubeCondLoad_VCS1 + VCS1excess + VCS1excessShell
		VCS1 += wire_VCS1_in - wire_VCS1_out


		VCS1_load = Rad_VCS1 + window_VCS1
		VCS1_load += cfact*(flexCondLoad3In + tubeCondLoad3In)
		VCS1_load += VCS1excess + VCS1excessShell
		VCS1_load += wire_VCS1_in




		VCS2 = Rad_VCS2  + window_VCS2 \
				-Rad_VCS1 - gasCoolingVCS2 - ocsCryocooler - MTexcess2 - VCS1excess
		VCS2 += flexCondLoad_VCS2 + tubeCondLoad_VCS2 - tubeCondLoad_MT
		VCS2 += VCS2excess


		VCS2_load = Rad_VCS2  + window_VCS2
		VCS2_load += cfact*(flexCondLoad4In + tubeCondLoad4In)
		VCS2_load += VCS2excess

		mdot = MT / l
		mdot2 = MTLoad / l

		#print VCS1, VCS2, mdot
		#Check if loads ~ zero

		if args.verbose:
			print('n={:d}'.format(n))

			print('Loadings')
			print('           |   SFT      |    MT      |   VCS1     |   VCS2     |')
			print('Aperture   | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
				.format(0.0, window_MT, window_VCS1, window_VCS2))
			print('Radiative  | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
				.format(Rad_SFT/SFTLoad, Rad_MT, Rad_VCS1, Rad_VCS2))
			print('Structural | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
				.format(cfact*flexCondLoad_SFT, cfact*flexCondLoad2In, cfact*flexCondLoad3In, cfact*flexCondLoad4In))
			print('Plumbing   | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
				.format(tubeCondLoad_SFT, tubeCondLoad_MT, cfact*tubeCondLoad3In, cfact*tubeCondLoad4In))
			print('Wiring     | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
				.format(0, wire_MT, wire_VCS1_in, 0))
			print('--------')
			print('Total      | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
			 	.format(SFTLoad, MTLoad, VCS1_load, VCS2_load))
			print('\n')



			print("VCS temp (K): T_VCS1: {:.1f}, T_VCS2: {:.1f}".format(T_VCS1, T_VCS2))
			print("VCS net loadings (W): VCS1: {:.3f}, VCS2: {:.3f}".format(VCS1, VCS2))
			print("\n\n")

		if (abs(VCS1) < eps and abs(VCS2) < eps):

			print('-------------------')
			print('VCS1cryocooler power: %1.3f W' % icsCryocooler)
			print('VCS2 cryocooler power: %1.3f W' % ocsCryocooler)

			print('--------')
			print('VCS1 gas cooling power: %1.3f W' % gasCoolingVCS1)
			print('VCS2 gas cooling power: %1.3f W' % gasCoolingVCS2)

			print('--------')
			print('mdot (g/s): %1.3f, Holdtime (days): %1.3f ' % (mdot, holdtime(mdot, numLiters = numLiters)))
			print('boil-off rate (L/day): %1.3f ' % (numLiters/holdtime(mdot, numLiters=numLiters)))

			print('SLPM (from MT)(L/min): %1.3f ' % (mdot2SLPM(mdot)))
			print('SLPM (from MTLoad)(L/min): %1.3f ' % (mdot2SLPM(mdot2)))

			print('MT power: %1.3f W' % (MT))
			print('MTLoad power: %1.3f W' % (MTLoad))
			print('--------')
			print('SFT Loading: %s' % uprint(SFTLoad))
			#print summary
			print('--------')
			print(' Stage | Temperature ')
			print(' VCS2  | %1.2f K ' % T_VCS2)
			print(' VCS1  | %1.2f K ' % T_VCS1)
			print('--------')
			if not args.noPrefix:
				def uprint8(fs):
					fout = ()
					for f in fs:
						fout = fout + (uprint(f, fmt='%8.2f'),)
					return fout

				print('Loads')
				print('           |   SFT      |    MT      |   VCS1     |   VCS2     |')
				print('Aperture   |%s |%s |%s |%s |' \
					% uprint8((0, window_MT, window_VCS1, window_VCS2)))
				print('Radiative  |%s |%s |%s |%s |' \
					% uprint8((Rad_SFT, Rad_MT, Rad_VCS1, Rad_VCS2)))
				# print('MLI       | %1.2e W | %1.2e W | %1.2e W |' % (0.0, mli_load_VCS1, mli_load_VCS2))
				print('Structural |%s |%s |%s |%s |' \
					% uprint8((cfact*flexCondLoad_SFT, cfact*flexCondLoad2In, \
					cfact*flexCondLoad3In, cfact*flexCondLoad4In)))
				print('Plumbing   |%s |%s |%s |%s |' \
					% uprint8((tubeCondLoad_SFT, tubeCondLoad_MT, \
					cfact*tubeCondLoad3In, cfact*tubeCondLoad4In)))
				print('Wiring     |%s |%s |%s |%s |'\
					% uprint8((0, wire_MT, wire_VCS1_in, 0)))
				print('--------')
				print('Total      |%s |%s |%s |%s |' \
				% uprint8((SFTLoad, MTLoad, VCS1_load, VCS2_load)))

				print('--------')
			else:
				print('Loads')
				print('           |   SFT      |    MT      |   VCS1     |   VCS2     |')
				print('Aperture   | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' \
					% (0, window_MT, window_VCS1, window_VCS2))
				print('Radiative  | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' \
					% (Rad_SFT, Rad_MT, Rad_VCS1, Rad_VCS2))
				# print('MLI       | %1.2e W | %1.2e W | %1.2e W |' % (0.0, mli_load_VCS1, mli_load_VCS2))
				print('Structural | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' \
					% (cfact*flexCondLoad_SFT, cfact*flexCondLoad2In, cfact*flexCondLoad3In, cfact*flexCondLoad4In))
				print('Plumbing   | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' \
					% (tubeCondLoad_SFT, tubeCondLoad_MT, cfact*tubeCondLoad3In, cfact*tubeCondLoad4In))
				print('Wiring     | {:1.2e} W | {:1.2e} W | {:1.2e} W | {:1.2e} W |'\
					.format(0, wire_MT, wire_VCS1_in, 0))
				print('--------')
				print('Total      | %1.2e W | %1.2e W | %1.2e W | %1.2e W |' % (SFTLoad, MTLoad, VCS1_load, VCS2_load))

				print('--------')
			print('Loads distribution')
			print('           |   SFT      |    MT      |   VCS1     |   VCS2     |')
			print('Aperture   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |'\
				.format(0.0, window_MT/MTLoad*100, window_VCS1/VCS1_load*100, window_VCS2/VCS2_load*100))
			print('Radiative  |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |'\
				.format(Rad_SFT/SFTLoad*100, Rad_MT/MTLoad*100, Rad_VCS1/VCS1_load*100, Rad_VCS2/VCS2_load*100))

			print('Structural |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |'\
				.format(cfact*flexCondLoad_SFT/SFTLoad*100, cfact*flexCondLoad2In/MTLoad*100, cfact*flexCondLoad3In/VCS1_load*100, cfact*flexCondLoad4In/VCS2_load*100))
			print('Plumbing   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |'\
				.format(tubeCondLoad_SFT/SFTLoad*100, tubeCondLoad_MT/MTLoad*100, cfact*tubeCondLoad3In/VCS1_load*100, cfact*tubeCondLoad4In/VCS2_load*100))
			print('Wiring     |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |  {:05.2f} %   |'\
				.format(0, wire_MT/MTLoad*100, wire_VCS1_in/VCS1_load*100, 0./VCS2_load*100))

			print('--------')

			print('In-band detector loading: %s' % uprint(inband))

			print('--------')
			print('Number of Iteration: {:d}\nRun time: {:.3f} s\neps: {:.3f} W'\
			.format(n, time()-T0, np.max(np.abs([VCS1,VCS2]))) )

			print('--------')

			return T_VCS1 , T_VCS2, mdot

		# # Cutting back on our precision
		# if ( n == np.floor(maxIter/2) ):
		# 	DeltaT = DeltaT/2
		# 	eps = eps*1.5
		# 	print(n)
		# if ( n == np.floor(0.75*maxIter) ):
		# 	DeltaT = DeltaT/2
		# 	eps = eps*1.5
		# 	print(n)

		n += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Thermal model')
	parser.add_argument('-ULDB', dest='ULDB', action='store_true', help='Run ULDB model instead of Theo?')
	parser.add_argument('-ULDB2', dest='ULDB2', action='store_true', help='Run ULDB *2* model instead of Theo?')
	parser.add_argument('-theo2', dest='theo2', action='store_true', help='Run theo model with large flexure intercept only to VCS2')
	parser.add_argument('-theo_alt1', dest='theo_alt1', action='store_true', help='Run theo model with as built dims, but connected only to VCS1 intercept')
	parser.add_argument('-lloro', dest='lloro', action='store_true', help='Theo model with large flexure intercept only to VCS1, equivalent to lloro model')
	parser.add_argument('-TNG', dest='TNG', action='store_true', help='Run TNG model')
	parser.add_argument('-TIM', dest='TIM', action='store_true', help='Run TIM model')
	parser.add_argument('-flexFact', dest = 'flexFactor', action = 'store', type=float, default=1.0, help='Reduction factor in flexure conduction')
	parser.add_argument('-ocsCoolers', dest = 'ocsCoolers', action = 'store', type = float, default = 0.0, help='Number of VCS2 coolers')
	parser.add_argument('-icsCoolers', dest = 'icsCoolers', action = 'store', type = float, default = 0.0, help='Number of VCS1 coolers')
	parser.add_argument('-mylarWindow', dest = 'mylarWindow', action = 'store_true', help='Use a 10-um Mylar window instead of the default 1/8" PE')
	parser.add_argument('-shaderWindow', dest = 'shaderWindow', action = 'store_true', help='Use a PE window with a 300K shader.')
	parser.add_argument('-keller', dest = 'keller', action = 'store_true', help='Use the keller MLI model')
	parser.add_argument('-sftEmpty', dest = 'sftEmpty', action = 'store_true', help='Model with the SFT empty')
	parser.add_argument('-noInserts', dest = 'windowsOpen', action = 'store_false', help='Run with no inserts installed (no filters)')
	parser.add_argument('-mtExcess', dest = 'mtExcess', action = 'store', type = float, default= 0.0, help='Excess load on MT, in Watts')
	parser.add_argument('-VCS1Excess', dest = 'VCS1Excess', action = 'store', type = float, default= 0.0, help='Excess load on VCS1, in Watts')
	parser.add_argument('-VCS2Excess', dest = 'VCS2Excess', action = 'store', type = float, default= 0.0, help='Excess load on VCS2, in Watts')
	parser.add_argument('-mtLLVCS1perc', dest = 'mtLLVCS1perc', action = 'store', type = float, default= 0.0, help='Percent of area light leak load on MT through VCS1')
	parser.add_argument('-mtLLVCS2perc', dest = 'mtLLVCS2perc', action = 'store', type = float, default= 0.0, help='Percent of area light leak load on MT through VCS2')
	parser.add_argument('-VCS1LLperc', dest = 'VCS1LLperc', action = 'store', type = float, default= 0.0, help='Percent of area light leak load on VCS1')
	parser.add_argument('-VVTemp', dest = 'VVTemp', action = 'store', type = float, default= 248.0, help='Vacuum vessel wall temperature')
	parser.add_argument('-ground', dest = 'ground', action = 'store_true', help='Ground loading?')
	parser.add_argument('-pins1', dest='pins1', action='store', type=float,
							default=1e-4, help='VCS1 interstitial pressure')
	parser.add_argument('-pins2', dest='pins2', action='store', type=float,
							default=1e-4, help='VCS2 interstitial pressure')
	parser.add_argument('-verbose', dest='verbose', action='store_true',
							help='Print step by step quantities.')
	parser.add_argument('-noPrefix', dest='noPrefix', action='store_true',
							help='Do not use prefix for output.')


	args = parser.parse_args()

	T_VCS1 , T_VCS2, mdot = find_equilibrium(args)
