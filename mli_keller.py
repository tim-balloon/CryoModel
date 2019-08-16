from numpy import *

sigma = 5.6705e-8 #[J/m2/s/K4] Stefan-Boltzmann

## constants for Lockheed Model
A_model=7.3e-8
B_model=7.07e-10
C_N2=1.46e4
C_He=4.89e4
m_N2=-0.48
m_He=-0.74
n=2.63


## emissivity
ep_al=0.04 # emissivity of MLI Al (possibly 0.04)
ep_ano=0.8 # emissivity of anodized Al
ep_au=0.02 # emissivity of Au plating

def P_rad_plates(ep_h, ep_c, Th, Tc):
	''' Inifine parallel plates.  Returns heat flux in [W/m^2]'''
	return sigma*(Th**4-Tc**4)/(1/ep_h+1/ep_c-1)

def P_rad_cylinder(ep_h, ep_c, Th, Tc, rh, rc):
	''' Infinite concentric cylinders.  Returns heat flux in [W/m^2]'''
	return sigma*(Th**4-Tc**4)/(1/ep_h+(rh/rc)*(1/ep_c-1))

def P_rad_MLI(N_l, Th, Tc, ep_r):
	'''MLI heat Flux from Keller model (eqn 4-18)
	returns q in W / m^2'''
	return B_model* (ep_r/N_l) * (Th**4.67-Tc**4.67)

def P_solid_MLI(N_l, N_s, Th, Tc):
	'''Solid conduction from Keller MLI model (eqn 4-18)
	N_l is # of layers, N_s is layers/cm
	Returns q in W / m^2'''
	return A_model*(N_s)**n/(N_l+1) * (Th+Tc)/2 * (Th-Tc)

def	p_ins(p, Th,Tc):
	return p*sqrt((Th+Tc)/2.0/300.0)

def	P_N2(p_ins, N_l, Th, Tc):
	return C_N2*p_ins/N_l * (Th**(m_N2+1) - Tc**(m_N2+1))

def P_He(p_ins, N_l, Th, Tc):
	return C_He*p_ins/N_l * (Th**(m_He+1) - Tc**(m_He+1))

def P_gas(p_ins, N_l, Th, Tc):
	return P_N2(p_ins, N_l, Th, Tc)+P_He(p_ins, N_l, Th, Tc)

def P_tot(p_ins, N_l, N_s, Th, Tc, e_r = ep_al):
	return P_solid_MLI(N_l, N_s, Th, Tc)+P_rad_MLI(N_l, Th, Tc, e_r)+P_gas(p_ins, N_l, Th, Tc)
