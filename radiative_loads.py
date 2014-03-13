#!/usr/bin/env python
# encoding: utf-8
"""
mli_conductivity.py

Created by Zigmund Kermish on 2014-01-20.  Heavily copy/pasted from Jon Gudmundsson's Matlab code, which was based
on Bill Jones' IDL code
"""

import sys
import os
import areas
import numpy as np
import mli_keller

#Stefan-Boltzmann constant
sigma = 5.6704E-12   #[J/s*cm^2*K^4] 


def mli_rad_keller(T_SFT, T_MT, T_VCS1, T_VCS2, T_Shell, 
	p_ins=1e-3, e_Al=0.15, alpha=0.15, beta=4.0e-3, config='theo'):
	'''returns the radiative loads INCLUDING all conductive and gas effects in MLI
	MLI is only used on VCS1 and VCS2 as gas loading would make MT MLI ineffective'''
	
	SFT_Area, MT_Area, VCS1_Area, VCS2_Area = areas.load_areas(config=config)
	#Thickness of mli sheets
	t1 = 2.00 #[cm]
	t2 = 3.81 #[cm]
	
	#number of layers
	N1 = 16
	N2 = 52
	N1_s = N1 / t1
	N2_s = N2 / t2
	
	Rad_VCS1 = mli_keller.P_tot(p_ins, N1, N1_s, T_VCS2, T_VCS1, e_r = e_Al)
	Rad_VCS2 = mli_keller.P_tot(p_ins, N2, N2_s, T_Shell, T_VCS2, e_r = e_Al)
	Rad_SFTtoMT = sigma*e_Al*(SFT_Area/2)*(T_MT**4-T_SFT**4)
	RadSFTtoVCS1 = sigma*e_Al*(SFT_Area/2)*(T_VCS1**4-T_SFT**4)
	
	Rad_MT = sigma*MT_Area*(T_VCS1**4-T_MT**4)/sum(1./effectEmiss(np.hstack((e_Al*0.8,
		MLIEmiss(T_MT,T_VCS1,0,alpha,beta), e_Al*0.9))))
	
	return Rad_SFTtoMT, RadSFTtoVCS1, Rad_MT, Rad_VCS1, Rad_VCS2

def mli_cond(T_VCS1,T_VCS2,T_Shell,Lambda = 1.0e-6, config='theo'):
	
	'''temperatures in Kelvin
		Lambda is the effective MLI conductivity, given in microWatts/cm/K
	'''
	
	SFT_Area, MT_Area, VCS1_Area, VCS2_Area = areas.load_areas(config=config)
	
	#Thickness of mli sheets
	t1 = 2.00 #[cm]
	t2 = 3.81 #[cm]
	
	lambda1 = 0.8*Lambda
	lambda2 = 1.2*Lambda
	
	mli_load_VCS1 = lambda1 * ( VCS1_Area / t1 ) * (T_VCS2-T_VCS1)
	mli_load_VCS2 = lambda2 * ( VCS2_Area / t2 ) * (T_Shell-T_VCS2)
	
	return mli_load_VCS1, mli_load_VCS2
	
def effectEmiss(eVector):
	L = len(eVector)
	if L < 2:
		print('emissivty vector input does not have enough elements')
		exit()
	eV1 = eVector[0:L-1]
	eV2 = eVector[1:L]
	EffEmm = (eV1*eV2)/(eV1+eV2-eV1*eV2)
	return EffEmm
	
def MLIEmiss(Tc, Th, N, alpha, beta):
	
	'''Assume a linear dependence of MLI layers on temperature and
	calculate array of emissivities with temperature dependence'''
	if N == 0:
		emissivities = np.array([])
	else:
		if Tc != Th:
			T = np.linspace(Tc, Th, N)
			emissivities = alpha + beta*T**0.5
		elif Tc == Th and N == 1:
			emissivities = alpha+beta*Tc**0.5
		else:
			emissivities = 0
	return emissivities
	
def rad_load(T_SFT, T_MT, T_VCS1,T_VCS2,T_Shell, e_Al=0.15, alpha=0.15, beta=4.0e-3, config = 'theo'):
	N2 = 0
	N3 = 16
	N4 = 52
		
	SFT_Area, MT_Area, VCS1_Area, VCS2_Area = areas.load_areas(config=config)
	
	#Radiative heat fluxess
	Rad_SFTtoMT = sigma*e_Al*SFT_Area/2*(T_MT**4-T_SFT**4)
	RadSFTtoVCS1 = sigma*e_Al*SFT_Area/2*(T_VCS1**4-T_SFT**4)
	
	Rad_MT = sigma*MT_Area*(T_VCS1**4-T_MT**4)/sum(1./effectEmiss(np.hstack((e_Al*0.8,
		MLIEmiss(T_MT,T_VCS1,N2,alpha,beta), e_Al*0.9))))
		
	Rad_VCS1 = sigma*VCS1_Area*(T_VCS2**4-T_VCS1**4)/sum(1./effectEmiss(np.hstack((e_Al*0.9,
		MLIEmiss(T_VCS1,T_VCS2,N3,alpha,beta), e_Al))))
		
	Rad_VCS2 = sigma*VCS2_Area*(T_Shell**4-T_VCS2**4)/sum(1./effectEmiss(np.hstack((e_Al, 
		MLIEmiss(T_VCS2,T_Shell,N4,alpha,beta), e_Al*1.1))))
		
	return Rad_SFTtoMT, RadSFTtoVCS1, Rad_MT, Rad_VCS1, Rad_VCS2
	
def main():
	
	mli_load_VCS1, mli_load_VCS2 = mli_cond(49.17, 170.31, 300)
	print('VCS1: %s W' % mli_load_VCS1)
	print('VCS2: %s W' % mli_load_VCS2)
	
	Rad_SFTtoMT, RadSFTtoVCS1, Rad_MT, Rad_VCS1, Rad_VCS2 = rad_load(T_SFT ,T_MT , T_VCS1 , T_VCS2, T_Shell,
		e_Al, alpha,beta)
	Rad_SFT = Rad_SFTtoMT+RadSFTtoVCS1
	


if __name__ == '__main__':
	main()

