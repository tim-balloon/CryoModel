#!/usr/bin/env python
# encoding: utf-8
"""
conductive_loads.py

Created by Zigmund Kermish on 2014-01-20.  Heavily copy/pasted from Jon Gudmundsson's Matlab code, which was based
on Bill Jones' IDL code
"""

import sys
import os
import scipy.integrate as integrate
import numpy as np

from gas_props import *

_this_dir, _this_filename = os.path.split(__file__)


def ss_low_cv(t_in):
	# low temperature conductivity stainless steel 316 in [W/m/K]
	# taken from M. Barucci, L. Lolli, L. Risegari, G. Ventura
	# Cryogenics 48 (2008) 166ñ168
	# see equation 3
	# only valid from 220 mK to 4.2 K,
	# see article for another fit for the conductivity at 40 - 220 mK

	A = 0.0556
	n = 1.15
	return A * t_in**n

T_SS_high, k_SS_high = np.loadtxt(_this_dir+'/thermalProp/ss304_cv.txt', unpack = True)
T_SS_low = np.arange(0.2, 3.9, 0.1)
k_SS_low = ss_low_cv(T_SS_low)
T_SS = np.append(T_SS_low, T_SS_high)
k_SS = np.append(k_SS_low, k_SS_high)

T_G10, k_G10 = np.loadtxt(_this_dir+'/thermalProp/g10_cv_combined.txt', unpack = True)

T_CF, k_CF = np.loadtxt(_this_dir+'/thermalProp/CF_cv.txt', unpack = True)


def llg10warp(t): return -2.64827 + 8.80228*t - 24.8998*t**2 + 41.1625*t**3 \
- 39.8754*t**4 + 23.1778*t**5 - 7.95635*t**6 + 1.48806*t**7 - .11701*t**8

def llg10norm(t): return -4.1236 + 13.788*t - 26.068*t**2 + 26.272*t**3 \
- 14.663*t**4 + 4.4954*t**5 - 0.6905*t**6 + 0.0397*t**7

def mfg_k(f):
	def g(t):
		return 10.0**f(np.log10(t))
	return g

g10warpk = mfg_k(llg10warp)
g10normk = mfg_k(llg10norm)
k_G10warp = g10warpk(T_G10)


def ringArea(OD, thickness):
	'''return the area of a ring'''
	return np.pi * ((OD/2.)**2 - (OD/2.-thickness)**2)

def HeatFlux_G10(T_min, T_max, AOverL, warp=False):
	'''Conductivity through G10 material
	AOverL: area/length	in meters
	return in Watts
	'''
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		if warp: K_G10 = k_G10warp
		else: K_G10 = k_G10

		return AOverL * integrate.trapz(np.interp(T, T_G10, K_G10), T)

def HeatFlux_CF(T_min, T_max, AOverL):
	'''Conductivity through Carbon Fiber
	AOverL: area/length	in meters
	return in Watts
	'''
	if T_min == T_max:
		return 0
	else:
		dT = min(T_CF[1:] - T_CF[:-1])
		T = np.arange(T_min, T_max, dT/2.)

		return AOverL * integrate.trapz(np.interp(T, T_CF, k_CF), T)

def HeatFlux_SS(T_min, T_max, AOverL):
	'''Conductivity through stainless steel
	AOverL: area/length	in meters
	return in Watts
	'''
	if T_min == T_max:
		return 0
	else:
		dT = min(T_SS[1:] - T_SS[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return AOverL * integrate.trapz(np.interp(T, T_SS, k_SS), T)

def HeatFlux_He(T_min, T_max, AOverL):
	'''Conductivity through Helium gas
	AOverL: area/length	in meters
	return in Watts
	'''

	if (abs(T_max-T_min) < 2.):
		return 0

	T = np.arange(T_min,T_max,0.01)

	return (AOverL)*integrate.trapz(helium_cv_ideal(T), x = T)

def TestLFlexH(T_min, T_max, L, T_G10, k_G10):
	'''Conductivity through large test cryostat flexures cross sectional area
	of length L'''
	t = 0.0009398 #[m] = .037 inches
	w = 0.0889 #[m] = 3.5 inches
	A = t*w;
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

def TestSFlexH(T_min, T_max, L, T_G10, k_G10):
	'''Conductivity through small test cryostat flexures cross sectional area
	of length L'''
	t = 0.000508 #[m] = .02 inches
	w = 0.0762 #[m] = 3 inches
	A = t*w;
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

def SFlexH(T_min, T_max, L, T_G10, k_G10):
	'''Conductivity through small Theo flexures cross sectional area
	of length L'''
	t = 0.00076
	w = 0.029# 0.01429 why did jon have half??
	A = t*w
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

def LFlexH(T_min, T_max, L, T_G10, k_G10):
	'''Conductivity through large Theo flexures cross sectional area
	of length L'''
	t = 0.00159    #[m]
	w = 0.127 #0.06513;    #[m]
	A = t*w
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

def AxFlexH(T_min, T_max, L, T_G10, k_G10):
	t = 0.00236;    #[m]
	w = 0.0127;    #[m]
	A = t*w;
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

def SFTFlex(T_min, T_max, L, T_G10, k_G10):
	t = 0.000787;    #[m]
	w = 0.03175;    #[m]
	A = t*w;
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

SSTthick = 2.0*0.000254   #[m]

def SSMTTube(T_min, T_max, L, T_SS, k_SS):
	OD = 0.01905    #[m]
	t = SSTthick    #[m]
	rout = OD/2
	rin = rout-t
	A = np.pi*(rout**2-rin**2) #[m^2]
	if T_min == T_max:
		return 0.0
	else:
		dT = min(T_SS[1:] - T_SS[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_SS, k_SS), x =T)

def SSSFTTube(T_min, T_max, L, T_SS, k_SS):
	OD = 0.0127    #[m]
	t = SSTthick #0.000254    #[m]
	rout = OD/2
	rin = rout-t
	A = np.pi*(rout**2-rin**2) #[m^2]
	if T_min == T_max:
		return 0.0
	else:
		dT = min(T_SS[1:] - T_SS[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_SS, k_SS), x = T)

def SSMTTubeGas(T_min, T_max, L):
	OD = 0.01905    #[m]
	t = SSTthick #0.000254    #[m]
	ID = OD-2*t
	A = np.pi*(ID)**2/4       #[m^2]

	if (abs(T_max-T_min) < 2.):
		return 0
	T = np.arange(T_min,T_max,0.01)

	return (A/L)*integrate.trapz(helium_cv_ideal(T), x = T)

def SSSFTTubeGas(T_min, T_max, L):
	OD = 0.0127    #[m]
	t = SSTthick #0.000254    #[m]
	ID = OD-2*t
	A = np.pi*(ID)**2/4       #[m^2]

	if (abs(T_max-T_min) < 2):
		return 0
	T = np.arange(T_min,T_max,0.01)

	return (A/L)*integrate.trapz(helium_cv_ideal(T), x = T)

def cond_loads(T1,T2,T3,T4,T5,sftPumped,sftEmpty,insNum, config = 'theo', flexFactor=1.0):

	'''
	--------------------------------------------------------------------------

	Notes:
	Calculating the conductive load through the vent and fill lines

	These calculations include both the conduction through the stainless steel
	tubes as well as the conduction through the stationary gas in the np.pipes
	that are normally valved off (since we are venting through the vcs vent
	line)

	The SFT fill and vent line are heat sunk at VCS2 and to some extent on
	the MT as well. There is OFHC strain relief on both the SFT fill and vent
	line that is roughly 0.6 meters away from the SFT and connected at the
	bottom of the main tank. We assume that this strain relief does not
	perform as an ideal heat sink and add a 5 K gradient between it and
	whatever the temperature of the main tank is. This will increase the load
	to the SFT somewhat.

	--------------------------------------------------------------------------
	'''
	if config != 'TNG' and config != 'TIM':

		L_MTFill_24 = 1.1
		L_MTFill_45 = 1.7

		L_MTVent_24 = 1.5
		L_MTVent_45 = 1.2

		L_SFTFill_12 = 0.8
		L_SFTFill_24 = 1.27
		L_SFTFill_45 = 1.524

		L_SFTVent_12 = 0.8
		L_SFTVent_24 = 1.27
		L_SFTVent_45 = 1.524

		#tubes are heat sunk to VCS2 (T4)

		MTFill24 = SSMTTube(T2,T4,L_MTFill_24,T_SS,k_SS)
		#print('MTFill metal tube: %1.4f' % MTFill24)
		MTFill24 += SSMTTubeGas(T2,T4,L_MTFill_24)
		#print('MTFill metal tube w/gas: %1.4f' % MTFill24)

		MTFill45 = SSMTTube(T4,T5,L_MTFill_45,T_SS,k_SS)
		MTFill45 += SSMTTubeGas(T4,T5,L_MTFill_45)

		MTVent24 = SSMTTube(T2,T4,L_MTVent_24,T_SS,k_SS)
		#print('MTVent metal tube: %1.4f' % MTVent24)
		MTVent24 += SSMTTubeGas(T2,T4,L_MTVent_24)
		#print('MTVent metal tube w/ gas: %1.4f' % MTVent24)

		MTVent45 = SSMTTube(T4,T5,L_MTVent_45,T_SS,k_SS)
		MTVent45 += SSMTTubeGas(T4,T5,L_MTVent_45)

		SFTFill12 = SSSFTTube(T1,T2,L_SFTFill_12,T_SS,k_SS)
		SFTVent12 = SSSFTTube(T1,T2,L_SFTVent_12,T_SS,k_SS)

		if sftPumped and not sftEmpty:
		  # If the SFT is pumped and not empty we assume that the loading through
		  # the stainless steel tubing will be conducted into the SFT rather than
		  # intercepted by the main tank. The theory being that the superfluid will
		  # creep up into the plumbing and intercept the heat...
			sftr = 0.0
			SFTFill24 = sftr*SSSFTTube(T2,T4,L_SFTFill_24,T_SS,k_SS)
			SFTVent24 = sftr*SSSFTTube(T2,T4,L_SFTVent_24,T_SS,k_SS)
			SFTFill12 = SFTFill12 + (1-sftr)*SSSFTTube(T2,T4,L_SFTFill_24,T_SS,k_SS)
			SFTVent12 = SFTVent12 + (1-sftr)*SSSFTTube(T2,T4,L_SFTVent_24,T_SS,k_SS)
		else:
		  # If the SFT is pumped but empty or not pumped and at 4.2 K we expect the
		  # copper strain relief to catch sftr of the heat load
			sftr = 1
			SFTFill24 = sftr*SSSFTTube(T2,T4,L_SFTFill_24,T_SS,k_SS)
			SFTVent24 = sftr*SSSFTTube(T2,T4,L_SFTVent_24,T_SS,k_SS)
			SFTFill12 = SFTFill12 + (1-sftr)*SSSFTTube(T2,T4,L_SFTFill_24,T_SS,k_SS)
			SFTVent12 = SFTVent12 + (1-sftr)*SSSFTTube(T2,T4,L_SFTVent_24,T_SS,k_SS)

		SFTFill45 = SSSFTTube(T4,T5,L_SFTFill_45,T_SS,k_SS)
		SFTVent45 = SSSFTTube(T4,T5,L_SFTVent_45,T_SS,k_SS)

		if not sftPumped:
			SFTFill12 = SFTFill12 + SSSFTTubeGas(T1,T2,L_SFTFill_12)
			SFTVent12 = SFTVent12 + SSSFTTubeGas(T1,T2,L_SFTVent_12)
			SFTFill24 = SFTFill24 + SSSFTTubeGas(T2,T4,L_SFTFill_24)
			SFTVent24 = SFTVent24 + SSSFTTubeGas(T2,T4,L_SFTVent_24)
			SFTFill45 = SFTFill45 + SSSFTTubeGas(T4,T5,L_SFTFill_45)
			SFTVent45 = SFTVent45 + SSSFTTubeGas(T4,T5,L_SFTVent_45)


		#--------------------------------------------------------------------------
		# SFT specific
		#--------------------------------------------------------------------------

		if not sftEmpty and sftPumped:
			insLoading = 300e-6 # Loading from 4K to the 1.5 K stage
		else:
			insLoading = 0

	else:

		if config =='TNG':
			sFactor45 = 1.
			sFactor34 = 1.
			sFactor23 = 1.

		else:
			sFactor45 = 4.
			sFactor34 = 4.
			sFactor23 = 4.

		# A over L ratio from BLAST excel
		ALHeFill45 = sFactor45* 8.58418E-05 #m
		ALHeFill34 = sFactor34* 7.24358E-05 #m
		ALHeFill23 = sFactor23* 9.20346E-05 #m
		ALHeFill12 = 1.86757E-05 #m

		ALHeFill45_gas = sFactor45* np.pi*(1./2)**2 /(2.9+3.5+0.5) * 0.0254 #m
		ALHeFill34_gas = sFactor34* np.pi*(.75/2)**2 /(3.925+2.625+0.5) * 0.0254 #m
		ALHeFill23_gas = sFactor23* np.pi*(.75/2)**2 /(0.5+2.625+0.5) * 0.0254 #m
		# ALHeFill12_gas = np.pi*(.5/2)**2 /(2.9+) * 0.0254 #m

		MTFill45 = HeatFlux_SS(T4,T5,ALHeFill45,) + HeatFlux_He(T4,T5,ALHeFill45_gas,)

		MTFill34 = HeatFlux_SS(T3,T4,ALHeFill34,) + HeatFlux_He(T3,T4,ALHeFill34,)
		MTFill23 = HeatFlux_SS(T2,T3,ALHeFill23,) + HeatFlux_He(T2,T3,ALHeFill23,)

		SFTPump12 = HeatFlux_SS(T1,T2,ALHeFill12,)

	#--------------------------------------------------------------------------
	# Calculating the final tubing loads
	#--------------------------------------------------------------------------
	if config == 'TNG' or config == 'TIM':
		tubeCondLoad1 = SFTPump12   # SFT

		#MT
		tubeCondLoad2in = MTFill23
		tubeCondLoad2out = -1*SFTPump12

		#VCS1
		tubeCondLoad3in = MTFill34
		tubeCondLoad3out = -1*MTFill23

		#VCS2
		tubeCondLoad4in  = MTFill45
		tubeCondLoad4out = -1*MTFill34

	else:
		#Adding load from tubing
		tubeCondLoad1 = (SFTFill12 + SFTVent12)  # SFT

		tubeCondLoad2in = (MTFill24 + MTVent24)+(SFTFill24 + SFTVent24) #MT
		tubeCondLoad2out = -1*(SFTFill12 + SFTVent12)

		tubeCondLoad3in = 0
		tubeCondLoad3out = 0

		tubeCondLoad4out = -1*tubeCondLoad2in
		tubeCondLoad4in  = MTFill45 + MTVent45 + SFTFill45 + SFTVent45



	#--------------------------------------------------------------------------
	# Calculating the conductive load through all sorts of structural elements
	#--------------------------------------------------------------------------


	#Calculating the total conductive load to MT, VCS1 and VCS2 from structural elements

	if config == 'theo':

		# Relevant lengths in meters
		L_MTLargeFlex = 0.0127  #0.5 inches
		L_VCS1LargeFlex = 0.0508  #2 inches
		L_VCS2LargeFlex = 0.0203  # 0.8 inches
		L_MTSmallFlex = 0.0330  #1.3 inches
		L_VCS2SmallFlex = 0.0330 # 1.3 inches
		L_MTAxFlex = 0.09015
		L_SFTFLex = 0.03937

		# Calculating heat loads for each junction
		#T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		LFlexToMT = LFlexH(T2,T3,L_MTLargeFlex,T_G10,k_G10) #MT -> VCS1
		SFlexToMT = SFlexH(T2,T3,L_MTSmallFlex,T_G10,k_G10) #MT-VCS1
		LFlexToVCS1 = LFlexH(T3,T4,L_VCS1LargeFlex,T_G10,k_G10) #VCS1->VCS2
		LFlexToVCS2 = LFlexH(T4,T5,L_VCS2LargeFlex,T_G10,k_G10) #VCS2->VV
		SFlexToVCS2 = SFlexH(T4,T5,L_VCS2SmallFlex,T_G10,k_G10) #VCS2-VV
		MTAxFlextoVCS1 = AxFlexH(T2,T3,L_MTAxFlex,T_G10,k_G10)
		if (T1 != T2):
			SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
		else:
			SFTFlexToMT = 0

		flexCondLoad1 = 7*SFTFlexToMT+insLoading*insNum
		flexCondLoad2in = 6*(LFlexToMT+SFlexToMT)+3*MTAxFlextoVCS1
		flexCondLoad2out = -7*SFTFlexToMT \
			-(SFTVent12+SFTFill12)
		flexCondLoad3in = 6*(LFlexToVCS1)
		flexCondLoad3out = -6*(LFlexToMT+SFlexToMT)-3*MTAxFlextoVCS1
		flexCondLoad4in = 6*(SFlexToVCS2+LFlexToVCS2)
		flexCondLoad4out = -6*LFlexToVCS1


	elif config == 'lloro':
		#vcs 1 intercept only
		# Relevant lengths in meters
		length = -0.03
		L_MTLargeFlex = 0.07874 + length  #full length to shell, no VCS intercepts
		L_VCS1LargeFlex = 0.0203 - length
		L_VCS2LargeFlex = 1e-4

		L_MTSmallFlex = 0.0330  #1.3 inches
		L_VCS2SmallFlex = 0.0330 # 1.3 inches
		L_MTAxFlex = 0.09015
		L_SFTFLex = 0.03937

		# Calculating heat loads for each junction
		#T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		LFlexToMT = LFlexH(T2,T3,L_MTLargeFlex,T_G10,k_G10)
		SFlexToMT = SFlexH(T2,T3,L_MTSmallFlex,T_G10,k_G10)
		LFlexToVCS1 = LFlexH(T3,T5,L_VCS1LargeFlex,T_G10,k_G10)
		#LFlexToVCS2 = LFlexH(T4,T5,L_VCS2LargeFlex,T_G10,k_G10)
		SFlexToVCS2 = SFlexH(T4,T5,L_VCS2SmallFlex,T_G10,k_G10)
		MTAxFlextoVCS1 = AxFlexH(T2,T3,L_MTAxFlex,T_G10,k_G10)
		if (T1 != T2):
			SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
		else:
			SFTFlexToMT = 0

		flexCondLoad1 = 7*SFTFlexToMT+insLoading*insNum
		flexCondLoad2in = 6*(LFlexToMT+SFlexToMT)+3*MTAxFlextoVCS1
		flexCondLoad2out = -7*SFTFlexToMT \
			-(SFTVent12+SFTFill12)
		flexCondLoad3in = 6*(LFlexToVCS1)
		flexCondLoad3out = -6*(LFlexToMT+ SFlexToMT)-3*MTAxFlextoVCS1
		flexCondLoad4in = 6*(SFlexToVCS2)
		flexCondLoad4out = -6*(LFlexToVCS1)

	elif config == 'theo2':
		#VCS2 intercept only
		# Relevant lengths in meters
		length = .00
		L_MTLargeFlex = 0.07874 + length  #full length to shell, no VCS intercepts
		#L_MTLargeFlex = 0.11 #4.5 inches, full length from MT -> VV
		L_VCS1LargeFlex = 1e-4
		#L_VCS2LargeFlex = 1e-4
		L_VCS2LargeFlex = 0.0203 - length

		L_MTSmallFlex = 0.0330  #1.3 inches
		L_VCS2SmallFlex = 0.0330 # 1.3 inches
		L_MTAxFlex = 0.09015
		L_SFTFLex = 0.03937

		# Calculating heat loads for each junction
		#T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		#LFlexToMT = LFlexH(T2,T5,L_MTLargeFlex,T_G10,k_G10)
		LFlexToMT = LFlexH(T2,T4,L_MTLargeFlex,T_G10,k_G10)
		SFlexToMT = SFlexH(T2,T3,L_MTSmallFlex,T_G10,k_G10)
		LFlexToVCS1 = 0.0*LFlexH(T3,T4,L_VCS1LargeFlex,T_G10,k_G10)
		LFlexToVCS2 = LFlexH(T4,T5,L_VCS2LargeFlex,T_G10,k_G10)
		SFlexToVCS2 = SFlexH(T4,T5,L_VCS2SmallFlex,T_G10,k_G10)
		MTAxFlextoVCS1 = AxFlexH(T2,T3,L_MTAxFlex,T_G10,k_G10)
		if (T1 != T2):
			SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
		else:
			SFTFlexToMT = 0

		flexCondLoad1 = 7*SFTFlexToMT+insLoading*insNum
		flexCondLoad2in = 6*(LFlexToMT+SFlexToMT)+3*MTAxFlextoVCS1
		flexCondLoad2out = -7*SFTFlexToMT \
			-(SFTVent12+SFTFill12)
		flexCondLoad3in = 0.0*(LFlexToVCS1)
		flexCondLoad3out = -6*(SFlexToMT)-3*MTAxFlextoVCS1
		flexCondLoad4in = 6*(SFlexToVCS2 + LFlexToVCS2)
		flexCondLoad4out = -6*(LFlexToMT)

	elif config == 'theo_alt1':
		#use as-built dimensions, but disconnect VCS2 flexure

		# Relevant lengths in meters
		L_MTLargeFlex = 0.0127  #0.5 inches MT -> VCS1
		L_VCS1LargeFlex = 0.0508  #2 inches VCS1 -> VCS2
		L_VCS2LargeFlex = 0.0203  # 0.8 inches
		L_VCS1LargeFlex += L_VCS2LargeFlex #add VCS2 length to VCS1 flexure, VCS1-> VV
		L_MTSmallFlex = 0.0330  #1.3 inches
		L_VCS2SmallFlex = 0.0330 # 1.3 inches
		L_MTAxFlex = 0.09015
		L_SFTFLex = 0.03937

		# Calculating heat loads for each junction
		#T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		LFlexToMT = LFlexH(T2,T3,L_MTLargeFlex,T_G10,k_G10) #MT -> VCS1
		SFlexToMT = SFlexH(T2,T3,L_MTSmallFlex,T_G10,k_G10) #MT-VCS1
		LFlexToVCS1 = LFlexH(T3,T5,L_VCS1LargeFlex,T_G10,k_G10) #VCS1->VCS2
		#LFlexToVCS2 = LFlexH(T4,T5,L_VCS2LargeFlex,T_G10,k_G10) #VCS2->VV
		SFlexToVCS2 = SFlexH(T4,T5,L_VCS2SmallFlex,T_G10,k_G10) #VCS2-VV
		MTAxFlextoVCS1 = AxFlexH(T2,T3,L_MTAxFlex,T_G10,k_G10)
		if (T1 != T2):
			SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
		else:
			SFTFlexToMT = 0

		flexCondLoad1 = 7*SFTFlexToMT+insLoading*insNum
		flexCondLoad2in = 6*(LFlexToMT+SFlexToMT)+3*MTAxFlextoVCS1
		flexCondLoad2out = -7*SFTFlexToMT \
			-(SFTVent12+SFTFill12)
		flexCondLoad3in = 6*(LFlexToVCS1) #goes to VV
		flexCondLoad3out = -6*(LFlexToMT+SFlexToMT)-3*MTAxFlextoVCS1
		flexCondLoad4in = 6*(SFlexToVCS2)
		flexCondLoad4out = 0.0

	elif config == 'ULDB':

		# Relevant lengths in meters
		L_VCS2toMTFlex = 0.03429  #1.35 inches
		L_VCS2Flex = 0.02921 # 1.15 inches
		L_VCS2toVCS1 = 0.0508 # 2 inches
		L_SFTFLex = 0.03937 #using theo SFT $s for now.

		# Calculating heat loads for each junction
		#T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		FlexToMT = TestLFlexH(T2,T4,L_VCS2toMTFlex,T_G10,k_G10) #MT -> VCS2
		FlexToVCS1 = TestSFlexH(T3,T4,L_VCS2toVCS1,T_G10,k_G10) #VCS1 -> VCS2
		FlexToVCS2 = TestLFlexH(T4,T5,L_VCS2Flex,T_G10,k_G10) #VCS2->VV

		if (T1 != T2):
			SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
		else:
			SFTFlexToMT = 0

		flexCondLoad1 = 3*SFTFlexToMT+insLoading*insNum
		flexCondLoad2in = 4*FlexToMT
		flexCondLoad2out = -3*SFTFlexToMT \
			-(SFTVent12+SFTFill12)
		flexCondLoad2in /= flexFactor #playing with improved flexures

		flexCondLoad3in = 4*FlexToVCS1
		flexCondLoad3out = 0 #VCS1 not conductively connected to any colder stages
		flexCondLoad4in = 4*FlexToVCS2
		flexCondLoad4out = -4*FlexToMT #VCS 2 -> MT connection

	elif config == 'ULDB2':

		# Relevant lengths in meters
		L_VCS1toMTFlex = 0.03429  #1.35 inches
		L_VCS2Flex = 0.02921 # 1.15 inches
		L_VCS2toVCS1 = 0.0508 # 2 inches
		L_SFTFLex = 0.03937 #using theo SFT $s for now.

		# Calculating heat loads for each junction
		#T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		FlexToMT = TestLFlexH(T2,T3,L_VCS1toMTFlex,T_G10,k_G10) #MT ->VCS1
		FlexToVCS1 = TestLFlexH(T3,T4,L_VCS2toVCS1,T_G10,k_G10) #VCS1 ->VCS2
		FlexToVCS2 = TestLFlexH(T4,T5,L_VCS2Flex,T_G10,k_G10)

		if (T1 != T2):
			SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
		else:
			SFTFlexToMT = 0

		flexCondLoad1 = 3*SFTFlexToMT+insLoading*insNum
		flexCondLoad2in = 4*FlexToMT
		flexCondLoad2out = -3*SFTFlexToMT \
			-(SFTVent12+SFTFill12)
		flexCondLoad2in /= flexFactor #playing with improved flexures

		flexCondLoad3in = 4*FlexToVCS1
		flexCondLoad3out = -4*FlexToMT #VCS1->MT
		flexCondLoad4in = 4*FlexToVCS2
		flexCondLoad4out = -4*FlexToVCS1 #VCS 2 -> VCS1 connection

	elif config == 'TNG':

		# Relevant lengths and areas in meters, from BLAST excel

		# supportive shells
		L_ShelltoVCS2 = 27.3125 * 0.0254
		A_ShelltoVCS2 = ringArea(35.625*0.0254, 0.0625*0.0254)

		L_VCS2toVCS1 = 12.625 * 0.0254
		A_VCS2toVCS1 = ringArea(32.125*0.0254, 0.0625*0.0254)

		L_VCS1toMT = 15.375 * 0.0254
		A_VCS1toMT = ringArea(30.63*0.0254, 0.04*0.0254)

		L_MTtoSFT = 3.8 * 0.0254
		A_MTtoSFT = ringArea(3.24*0.0254, 0.02*0.0254)


		# Calculating heat loads for each junction
		# T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		warp = True

		FluxToVCS2 = HeatFlux_G10(T4, T5, A_ShelltoVCS2/L_ShelltoVCS2, warp=warp)
		FluxToVCS1 = HeatFlux_G10(T3,T4,A_VCS2toVCS1/L_VCS2toVCS1, warp=warp) 	#VCS1 ->VCS2
		FluxToMT = HeatFlux_G10(T2,T3,A_VCS1toMT/L_VCS1toMT, warp=warp) 		#MT ->VCS1

		if (T1 != T2):
			FluxToSFT = HeatFlux_G10(T1,T2, A_MTtoSFT/L_MTtoSFT, warp=warp)
		else:
			FluxToSFT = 0

		# SFT
		flexCondLoad1 = FluxToSFT

		# MT
		flexCondLoad2in = FluxToMT
		flexCondLoad2out = -1*FluxToSFT
		# flexCondLoad2in /= flexFactor #playing with improved flexures

		# VCS1
		flexCondLoad3in = FluxToVCS1
		flexCondLoad3out = -1*FluxToMT #VCS1->MT

		# VCS2
		flexCondLoad4in = FluxToVCS2
		flexCondLoad4out = -1*FluxToVCS1 #VCS 2 -> VCS1 connection

	elif config == 'TIM':

		# Relevant lengths and areas in meters, from BLAST excel

		# supportive shells
		L_ShelltoVCS2 = 27.3125 * 0.0254
		A_ShelltoVCS2 = ringArea(35.625*0.0254, 0.0625*0.0254)

		L_VCS2toVCS1 = 12.625 * 0.0254
		A_VCS2toVCS1 = ringArea(32.125*0.0254, 0.0625*0.0254)

		L_VCS1toMT = 15.375 * 0.0254
		A_VCS1toMT = ringArea(30.63*0.0254, 0.04*0.0254)

		L_MTtoSFT = 3.8 * 0.0254
		A_MTtoSFT = ringArea(3.24*0.0254, 0.02*0.0254)

		L_MTtoSFT_Truss = 0.2
		A_MTtoSFT_Truss = np.pi *1.0**2 *1e-4 # area of OD =2cm truss
		n_truss = 8

		# Calculating heat loads for each junction
		# T1 = SFT, T2 = 4k, T3 = VCS1, T4 = VCS2, T5 = 300K
		warp = True

		FluxToVCS2 = HeatFlux_G10(T4, T5, A_ShelltoVCS2/L_ShelltoVCS2, warp=warp)
		FluxToVCS1 = HeatFlux_G10(T3,T4,A_VCS2toVCS1/L_VCS2toVCS1, warp=warp) 	#VCS1 ->VCS2
		FluxToMT = HeatFlux_G10(T2,T3,A_VCS1toMT/L_VCS1toMT, warp=warp) 		#MT ->VCS1

		if (T1 != T2):
			FluxToSFT = HeatFlux_G10(T1,T2, A_MTtoSFT/L_MTtoSFT, warp=warp)
			FluxToSFT += HeatFlux_CF(T1,T2, n_truss* A_MTtoSFT_Truss/L_MTtoSFT_Truss,)
		else:
			FluxToSFT = 0

		# SFT
		flexCondLoad1 = FluxToSFT

		# MT
		flexCondLoad2in = FluxToMT
		flexCondLoad2out = -1*FluxToSFT
		# flexCondLoad2in /= flexFactor #playing with improved flexures

		# VCS1
		flexCondLoad3in = FluxToVCS1
		flexCondLoad3out = -1*FluxToMT #VCS1->MT

		# VCS2
		flexCondLoad4in = FluxToVCS2
		flexCondLoad4out = -1*FluxToVCS1 #VCS 2 -> VCS1 connection





	return (tubeCondLoad1, tubeCondLoad2in, tubeCondLoad2out, tubeCondLoad3in,
		tubeCondLoad3out, tubeCondLoad4in, tubeCondLoad4out,
	    flexCondLoad1, flexCondLoad2in, flexCondLoad2out, flexCondLoad3in,
	    flexCondLoad3out, flexCondLoad4in, flexCondLoad4out)






def main():
	pass


if __name__ == '__main__':
	main()
