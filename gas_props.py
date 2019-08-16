#!/usr/bin/env python
# encoding: utf-8
"""
gas_props.py

Created by Zigmund Kermish on 2014-01-20.  Heavily copy/pasted from Jon Gudmundsson's Matlab code, which was based
on Bill Jones' IDL code
"""

import sys
import os
import numpy as np
import scipy.integrate as integrate

T_He, Cp_He = np.loadtxt('thermalProp/helium_cp.txt', unpack = True)

def mdot2SLPM(mdot):
	'''
	Converting from kg/s to standard liters per minute (SLPM)
	'''

	HeDensAtBP = 124.98; #kg/m^3
	HeDensATStp = 0.1786; #g/L

	SLPM = 60 * ( mdot / HeDensATStp );
	return SLPM

def SLPM2mdot(SLPM):
	'''Converting from SLPM to kg/s'''


	HeDensAtBP = 124.98; #kg/m^3
	HeDensATStp = 0.1786; #g/L

	mdot = HeDensATStp*SLPM / 60.
	return mdot

def holdtime(mdot, numLiters = 1000):
	'''Calculate the hold time of the cryostat'''

	He_density = 125 #[g/l]
	#Seconds in a day
	spd = 24*3600;    #[s/day]
	days = numLiters*He_density/mdot/spd;
	return days

def CpInt(T_min, T_max, T_gas, Cp_gas):
	'''Intergration of specific heat'''

	dT = min(T_gas[1:] - T_gas[:-1])
	T = np.arange(T_min, T_max, dT/2.)
	CpInt = integrate.trapz(np.interp(T, T_gas, Cp_gas), T)
	return CpInt

def helium_lambda(T):
	'''
	Helium conductivity from pg 16 of "handbook of thermal conductivity of liquids and gases"
	functional fit to tabulated data across range from 2.2 to 6000K at 0.1MPa = 1atm
	returns conductivity in Watts/(m*K)
	'''

	return (2.81*T**0.7 - 9.5*T**(-1) + 3 + 3.1*10**(-3)*T+2.9*10**(-7)*T**2)*10**(-3) # W/(m*K)

def helium_cv_ideal(T):
	'''
	helium_cv - conduction of helium at 1 atm based on the following
	equation for the conduction of ideal gas:
	k = (1/3)*c*rho*<v>*mfp

	kout = helium_cv(t_in)
	'''

	m_He = 4.002602*1.660538e-27 #[kg]
	k_B = 1.3806503e-23 #[J/K]
	p_gauge = 2.
	p_psi = 14.7 #[psi]
	p_atm = ((p_gauge+p_psi)/(p_psi))*1.013e5 #[Pa]
	d = 62e-12 #[m]

	rho = he_rho(T)
	v = np.sqrt(3.*k_B*T/m_He)
	mfp = (k_B*T)/(np.sqrt(2)*np.pi*d**2*p_atm)
	cp = helium_cp(T)

	kout = (1./3.)*cp*rho*v*mfp
	return kout

def he_rho(t_in, phase = 'gas'):
	''' Get Helium density using interpolation
	'''

	if phase == 'gas':
		t = np.array([4.2, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300])
		ro = np.array([16.71, 12.07, 5.05, 2.45, 1.625, 1.218, 0.974, 0.641, 0.487, 0.390,
			0.325, 0.278, 0.244, 0.217, 0.195, 0.1773, 0.1625])
	elif phase =='liquid':
		t = np.array([0.1, 1.8, 2.2, 2.5, 3.0, 4.0, 4.2, 4.5, 4.9])
		ro = np.array([1.4514, 1.45308, 1.461049, 1.448402,
			1.412269, 1.289745, 1.254075, 1.188552, 1.060033])*1e2; #g/L
	return np.interp(t_in, t, ro)

def helium_cp(t_in, phase = 'gas'):
	'''
	helium_cp - %above 4.2 K specific heat of helium gas at 1atm, [J/kg/K]
	below 4.2 K specific heat of liquid at svp [J/kg/K].

	c = helium_cp(t_in)
	'''
	
	if phase == 'gas':
		t0 = np.array([4.2, 5, 10, 20, 30, 40, 50, 300])
		c0 = 1e3*np.array([7.07, 6.32, 5.43, 5.25, 5.23, 5.21, 5.20, 5.20])
	elif phase == 'liquid':
		t0 = np.array([2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
	 					3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
		c0 = np.array([2.65, 2.45, 2.4, 2.3, 2.32, 2.4, 2.45, 2.5, 2.6, 2.7, 2.8, 2.95, 3.1,
	    		3.25, 3.45, 3.65, 3.8, 4, 4.25, 4.5, 4.8, 5.1, 5.5, 6.0, 6.7, 7.5,
	    		8.6, 11])*1e3
	return np.interp(t_in, t0, c0)

def main():
	pass


if __name__ == '__main__':
	main()
