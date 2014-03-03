#!/usr/bin/env python
# encoding: utf-8
"""
conductive_loads.py

Created by Zigmund Kermish on 2014-01-20.  Heavily copy/pasted from Jon Gudmundsson's Matlab code, which was based
on Bill Jones' IDL code"""

import sys
import os
import scipy.integrate as integrate
import numpy as np

from gas_props import *

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

T_SS_high, k_SS_high = np.loadtxt('thermalProp/ss304_cv.txt', unpack = True)
T_SS_low = np.arange(0.2, 3.9, 0.1)
k_SS_low = ss_low_cv(T_SS_low)
T_SS = np.append(T_SS_low, T_SS_high)
k_SS = np.append(k_SS_low, k_SS_high)

T_G10, k_G10 = np.loadtxt('thermalProp/g10_cv_combined.txt', unpack = True)

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
	t = 0.00076;
	w = 0.01429;
	A = t*w;
	if T_min == T_max:
		return 0
	else:
		dT = min(T_G10[1:] - T_G10[:-1])
		T = np.arange(T_min, T_max, dT/2.)
		return (A/L) * integrate.trapz(np.interp(T, T_G10, k_G10), T)

def LFlexH(T_min, T_max, L, T_G10, k_G10):
	'''Conductivity through large Theo flexures cross sectional area
	of length L'''
	t = 0.00159;    #[m]
	w = 0.06513;    #[m]
	A = t*w;
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

def SSMTTube(T_min, T_max, L, T_SS, k_SS):
	OD = 0.01905    #[m]
	t = 0.000254    #[m]

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
	t = 0.000254    #[m]
	
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
	t = 0.000254    #[m]
	ID = OD-2*t
	A = np.pi*(ID)**2/4       #[m^2]

	if (abs(T_max-T_min) < 2.):
		return 0
	T = np.arange(T_min,T_max,0.01)

	return (A/L)*integrate.trapz(helium_cv_ideal(T), x = T)

def SSSFTTubeGas(T_min, T_max, L):
	OD = 0.0127    #[m]
	t = 0.000254   #[m]
	ID = OD-2*t
	A = np.pi*(ID)**2/4       #[m^2]

	if (abs(T_max-T_min) < 2):
		return 0
	T = np.arange(T_min,T_max,0.01)

	return (A/L)*integrate.trapz(helium_cv_ideal(T), x = T)
	
def cond_loads(T1,T2,T3,T4,T5,sftPumped,sftEmpty,insNum, config = 'theo'):
	
	#--------------------------------------------------------------------------
	# Notes:
	# Calculating the conductive load through the vent and fill lines, these
	# calculations include both the conduction through the stainless steel
	# tubes as well as the conduction through the stationary gas in the np.pipes
	# that are normally valved off (since we are venting through the vcs vent
	# line)
	#
	# The SFT fill and vent line are heat sunk at VCS2 and to some extent on
	# the MT as well. There is OFHC strain relief on both the SFT fill and vent
	# line that is roughly 0.6 meters away from the SFT and connected at the
	# bottom of the main tank. We assume that this strain relief does not
	# perform as an ideal heat sink and add a 5 K gradient between it and
	# whatever the temperature of the main tank is. This will increase the load
	# to the SFT somewhat.
	#
	#--------------------------------------------------------------------------

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
		SFTFill12 = SFTFill12 + SSSFTTube(T2,T4,L_SFTFill_24,T_SS,k_SS)
		SFTVent12 = SFTVent12 + SSSFTTube(T2,T4,L_SFTVent_24,T_SS,k_SS)
		SFTFill24 = 0 
		SFTVent24 = 0
	else:
	  # If the SFT is pumped but empty or not pumped and at 4.2 K we expect the  
	  # copper strain relief to catch sftr of the heat load
		sftr = 0.5
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

	#--------------------------------------------------------------------------
	# SFT specific
	#--------------------------------------------------------------------------
	if not sftEmpty and sftPumped:
	  insLoading = 300e-6 # Loading from 4K to the 1.5 K stage
	else:
	  insLoading = 0
	#--------------------------------------------------------------------------

	#--------------------------------------------------------------------------
	# Calculating the conductive load through all sorts of flexures
	#--------------------------------------------------------------------------

	# Relevant lengths in meters
	L_MTLargeFlex = 0.0127  #0.5 inches
	L_VCS1LargeFlex = 0.0508  #2 inches
	L_VCS2LargeFlex = 0.0203  # 0.8 inches
	L_MTSmallFlex = 0.0330  #1.3 inches
	L_VCS2SmallFlex = 0.0330 # 1.3 inches
	L_MTAxFlex = 0.09015
	L_SFTFLex = 0.03937

	# Calculating heat loads for each junction
	LFlexToMT = LFlexH(T2,T3,L_MTLargeFlex,T_G10,k_G10)
	SFlexToMT = SFlexH(T2,T3,L_MTSmallFlex,T_G10,k_G10)
	LFlexToVCS1 = LFlexH(T3,T4,L_VCS1LargeFlex,T_G10,k_G10)
	LFlexToVCS2 = LFlexH(T4,T5,L_VCS2LargeFlex,T_G10,k_G10)
	SFlexToVCS2 = SFlexH(T4,T5,L_VCS2SmallFlex,T_G10,k_G10)
	MTAxFlextoVCS1 = AxFlexH(T2,T3,L_MTAxFlex,T_G10,k_G10)
	if (T1 != T2):
		SFTFlexToMT = SFTFlex(T1,T2,L_SFTFLex,T_G10,k_G10)
	else:
		SFTFlexToMT = 0

	#--------------------------------------------------------------------------

	#--------------------------------------------------------------------------
	# Making the final calculations
	#--------------------------------------------------------------------------

	#Calculating the total conductive load to MT, VCS1 and VCS2 from flexures
	if config == 'theo':
		flexCondLoad1 = 7*SFTFlexToMT+insLoading*insNum
		flexCondLoad2 = 6*(LFlexToMT+SFlexToMT)+3*MTAxFlextoVCS1-7*SFTFlexToMT \
		  -(SFTVent12+SFTFill12)
		flexCondLoad3in = 6*(LFlexToVCS1)
		flexCondLoad3out = -6*(LFlexToMT+SFlexToMT)-3*MTAxFlextoVCS1
		flexCondLoad4in = 6*(SFlexToVCS2+LFlexToVCS2)
		flexCondLoad4out = -6*LFlexToVCS1
	elif config == 'ULDB':
		flexCondLoad1 = 3*SFTFlexToMT+insLoading*insNum
		flexCondLoad2 = 2*(LFlexToMT+SFlexToMT)-3*SFTFlexToMT \
		  -(SFTVent12+SFTFill12)
		flexCondLoad3in = 2*(LFlexToVCS1)
		flexCondLoad3out = -2*(LFlexToMT+SFlexToMT)
		flexCondLoad4in = 2*(SFlexToVCS2+LFlexToVCS2)
		flexCondLoad4out = -2*LFlexToVCS1
		

	#Adding load from tubing
	tubeCondLoad1 = (SFTFill12 + SFTVent12)
	tubeCondLoad2 =  (MTFill24 + MTVent24)+(SFTFill24 + SFTVent24)
	tubeCondLoad4out = -1*tubeCondLoad2
	tubeCondLoad4in  = MTFill45 + MTVent45 + SFTFill45 + SFTVent45
	
	return (tubeCondLoad1, tubeCondLoad2, tubeCondLoad4in, tubeCondLoad4out, 
	    flexCondLoad1, flexCondLoad2, flexCondLoad3in, 
	    flexCondLoad3out, flexCondLoad4in, flexCondLoad4out)
	

	#--------------------------------------------------------------------------

def main():
	pass


if __name__ == '__main__':
	main()

