import numpy as np
from scipy.interpolate import interp1d

def d2awg(d, clip=True):
	arg = ((np.log(d)-np.log(0.000127))*39./np.log(92.))/2.
	if clip:
		arg = np.round(arg)
	return 36 - arg*2

def a2awg(a):
	return d2awg(np.sqrt(a/np.pi)*2)

def awg2d(awg):
	return 0.000127*92.0**((36.0-awg)/39.0)

def awg2a(awg):
	return np.pi*awg2d(awg)**2/4.0

def pbronze_cv(tin):
	# Phosphor Bronze thermal conductivity [W/m-K]
	t=[4,10.0,20.0,80.0,150.0,300.0]
	k=[1.6,4.6,10.0,25.0,34.0,48.0]
	f = interp1d(t,k,'quadratic', bounds_error=False,fill_value=1.6)
	return f(tin)

def manganin_cv(tin):
	t=[4.0,10.0,20.0,80.0,150.0,300.0]
	k=[0.5,2.0,3.3,13.0,16.0,22.0]
	f = interp1d(t,k,'quadratic', bounds_error=False, fill_value=0.5)
	return f(tin)

def copper_cv(tin, RRR=50):
	# assumes RRR=50
	t=[4.0,10.0,20.0,80.0,150.0,300.0]
	# k=[9.0,30.0,70.0,300.0,700.0,1100.0,600.0,410.0,400.0]
	if RRR == 100:
		k = [638.0, 1540.0, 2423.0, 529.0, 418.0, 396.0] # RRR = 100
	elif RRR == 50:
		k = [318.0,778.0,1368.0,500.0,408.0,392.0] # RRR = 50
	f = interp1d(t,k,'quadratic', bounds_error=False, fill_value=300)

	return f(tin)

def al6061_cv(tin):
	# Al 6061-T6 thermal conductivity [W/m-K]
	tin = np.atleast_1d(tin)
	a=[0.07981,1.0957,-0.07277,0.08084,0.02803,-0.09464,0.04179,-0.00571,0]
	out = np.ones(len(tin))*a[0]
	for i in range(1,len(a)):
		out += a[i] * np.log10(tin)**i
	out = 10.0**out

	# scale linearly for electrons
	Tpiv = 5
	p = np.where(tin < Tpiv)[0]
	if not len(p):
		return out.squeeze()
	tref = al6061_cv(Tpiv)
	out[p] = tref * (tin[p]/Tpiv)
	return out.squeeze()

def alum_rho(tin):
	# from http://www.nist.gov/data/PDFfiles/jpcrd260.pdf
	t = [1, 2, 4, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
		 150, 200, 250, 273, 293, 300]
	r = [0.0001, 0.000102, 0.000109, 0.000139, 0.000193, 0.000346,
		 0.000755, 0.00187, 0.00453, 0.0181, 0.0478, 0.0959, 0.1624,
		 0.245, 0.339, 0.442, 1.006, 1.587, 2.157, 2.417, 2.650, 2.733]
	f = interp1d(t, r, 'quadratic')
	return f(tin)

def pbronze_rho(tin):
	t = [4.2, 77, 305]
	r = np.array([8.56, 8.83, 10.3]) * awg2a(36)
	f = interp1d(t,r,'quadratic')
	return f(tin)

def manganin_rho(tin):
	t = [4.2, 77, 305]
	r = np.array([34.6, 36.5, 38.8]) * awg2a(36)
	f = interp1d(t,r,'quadratic')
	return f(tin)

# def copper_rho(tin):
#     t = [4.2, 77, 305]
#     r = np.array([0.003, 0.04, 0.32])*awg2a(30)
#     f = interp1d(t,r,'quadratic')
#     return f(tin)

# def pbronze_rho_wf(tin):
#     L0 = 2.445e-8
#     return L0 * np.asarray(tin) / pbronze_cv(tin)

# def manganin_rho_wf(tin):
#     L0 = 2.445e-8
#     return L0 * np.asarray(tin) / manganin_cv(tin)

def copper_rho(tin):
	L0 = 2.445e-8 # W/m/K^2
	return L0 * np.asarray(tin) / copper_cv(tin)

def pbronze_r(tin,gauge=36):
	return pbronze_rho(tin) / awg2a(gauge)

def manganin_r(tin,gauge=36):
	return manganin_rho(tin) / awg2a(gauge)

def copper_r(tin,gauge=30):
	return copper_rho(tin) / awg2a(gauge)

def load_func(func, hstemps, lengths, aol):
	tmin = np.min(hstemps)
	# tin = np.linspace(np.min(hstemps),np.max(hstemps),100)
	# P = 0
	P = np.zeros(len(lengths)-1)
	for ii,length in enumerate(lengths[1:]):
		tarr = np.linspace(tmin, hstemps[ii+1], 100)
		karr = func(tarr)
		aol1 =  aol / (length - lengths[ii])
		P[ii] = np.trapz(karr, tarr) * aol1
	return P

def load_funci(kfunc, rfunc, hstemps, lengths, aol, current):
	tmin = np.min(hstemps)
	P = np.zeros(len(lengths)-1)
	for ii, length in enumerate(lengths[1:]):
		tmax = hstemps[ii+1]
		tarr = np.linspace(tmin, hstemps[ii+1], 100)
		karr = kfunc(tarr)
		rarr = rfunc(tarr)
		aol1 =  aol / (length - lengths[ii])
		if tmax == tmin:
			P[ii] = current**2 * np.mean(rarr) / aol1
		else:
			rk = np.trapz(karr * rarr, tarr) / (tmax - tmin)
			rav = np.trapz(rarr, tarr) / (tmax - tmin)
			if not current:
				P[ii] = np.trapz(karr, tarr) * aol1
			else:
				d = current**2 / 2 * rav**2 / rk / aol1**2
				c = ((tmax - tmin)**2 + 2 * d * (tmax + tmin) + d**2) / 4 / d
				P[ii] = current * np.sqrt(2 * rk * (c - tmin))
	return P

def wire_info(material, length, gauge=None, Rtarget=None, Npins=1,
			  N=1, sink_points={'mt':0,'vv_mt':1}, current=None):
	if material == 'manganin':
		rfunc = manganin_rho
		kfunc = manganin_cv
		gauge0 = 36
	elif material == 'pbronze':
		rfunc = pbronze_rho
		kfunc = pbronze_cv
		gauge0 = 36
	elif material == 'copper':
		rfunc = copper_rho
		kfunc = copper_cv
		gauge0 = 30
	elif material == 'aluminum':
		rfunc = alum_rho
		kfunc = al6061_cv
		gauge0 = 10

	Tarr = np.linspace(4.2,300,100)
	dT = Tarr[-1]-Tarr[0]

	if gauge is not None:
		if gauge == 0:
			gauge = gauge0;
		area = awg2a(gauge)
		aol = area / length
		R = np.trapz(rfunc(Tarr), Tarr) / dT / aol # resistance, Ohms
		# print gauge, area, aol
		if Rtarget is not None:
			N = np.int(np.ceil(R / Rtarget))
	else:
		area0 = awg2a(gauge0)
		aol0 = area0 / length
		R0 = np.trapz(rfunc(Tarr), Tarr) / dT / aol0
		gauge = a2awg(R0/Rtarget * awg2a(gauge0))
		area = awg2a(gauge)
		aol = area / length
		R = R0 * area0 / area

	# H = 1000.0*2590.6 # enthalpy of vaporization, mJ/L
	# Tflight = 86400*30 # length of flight

	def calc_load(tdict):
		keys = sorted(sink_points.keys(), key=lambda x: sink_points[x])
		tvec = [tdict[k.split('_')[0]] for k in keys]
		lvec = [sink_points[k] for k in keys]
		pvec = load_func(kfunc, tvec, lvec, aol * N) * float(Npins) # watts
		pdict = dict()
		for idx,k in enumerate(keys[1:]):
			if '_' not in k: continue
			pdict[k] = pvec[idx]
		return pdict

	def calc_loadi(tdict):
		keys = sorted(sink_points.keys(), key=lambda x: sink_points[x])
		tvec = [tdict[k.split('_')[0]] for k in keys]
		lvec = [sink_points[k] for k in keys]
		pvec = load_funci(kfunc, rfunc, tvec, lvec, aol * N, current) * float(Npins)
		pdict = dict()
		for idx, k in enumerate(keys[1:]):
			if '_' not in k: continue
			pdict[k] = pvec[idx]
		return pdict

	info = {'mat':material,'R':R/N,'awg':gauge,'N':N,'Npins':Npins,'P':calc_load,
			'sink':sink_points, 'I':current, 'PI': calc_loadi}
	return info
	# L = P*Tflight/H

def shield_info(material, length, width, thickness=0.005):
	if material == 'aluminum':
		kfunc = al6061_cv
		thick0 = 0.005
	elif material == 'mylar':
		kfunc = al6061_cv
		thick0 = 0.001

# wiring parameters
length_hk = 5*12.0*0.0254 # length of cryogenically active wiring, m
sink_points_hk = {'mt': 0, 'vcs1_mt': 0.4, 'vv_vcs1': 1}

length_mce = 3 # length of cryogenically active wiring, m
sink_points_mce = {'mt': 0, 'vcs1_mt': 0.2, 'vv_vcs1': 1}

length_cop = 10*12.0*0.0254
sink_points_cop = {'mt': 0, 'mt1_mt': 1}

mce_gauge = d2awg(0.1524e-3, clip=False) # 38 SWG

# wire material for each component
Winfo = {'HKd':wire_info('manganin', length_hk, gauge=36, Npins=12*2,
							   sink_points=sink_points_hk, current=10e-6),
		 'HKn':wire_info('manganin', length_hk, gauge=36, Npins=8*2,
						 sink_points=sink_points_hk, current=2e-9),
		 'HKr':wire_info('manganin', length_hk, gauge=36, Npins=8*2,
						 sink_points=sink_points_hk, current=3e-8),
		 'Pump':wire_info('manganin', length_hk, gauge=36, Npins=2, N=2,
						  sink_points=sink_points_hk, current=0.1),
		 'Hsw':wire_info('manganin', length_hk, gauge=36, Npins=2, N=2,
						  sink_points=sink_points_hk, current=5e-4),
		 'HKh':wire_info('manganin', length_hk, gauge=36, Npins=10,
						 sink_points=sink_points_hk, current=0),
		 'HK_spare':wire_info('manganin', length_hk, gauge=36, Npins=26,
							  sink_points=sink_points_hk, current=0),
		 # 'HK':wire_info('manganin', length_hk, gauge=36, Npins=50*2,
		 #                sink_points=sink_points_hk),
		 'HKth':wire_info('copper', length_hk, gauge=30, Npins=4*2 / 6.,
						  sink_points=sink_points_hk, current=0),
		 'HKtd':wire_info('manganin', length_hk, gauge=36, Npins=10 * 2 / 6.,
						  sink_points=sink_points_hk, current=10e-6),
		 # 'MCEtes':wire_info('manganin', length_mce, gauge=34.45, Npins=16*2,
		 #                    sink_points=sink_points_mce, current=0.04/1.11e3),
		 # 'MCEfb':wire_info('manganin', length_mce, gauge=34.45, Npins=33*2,
		 #                   sink_points=sink_points_mce, current=8*10e-6/15.2e3),
		 'MCE':wire_info('manganin', length_mce, gauge=34.45, Npins=3*100, #-(33+16)*2,
						 sink_points=sink_points_mce),
		 'MCEsh':wire_info('aluminum', length_mce, gauge=40, Npins=3,
						   sink_points=sink_points_mce),
		 'HKsh':wire_info('aluminum', length_hk, gauge=43, Npins=2,
						  sink_points=sink_points_hk),
		 # 'MCEdr':wire_info('copper', length_mce, gauge=38, Npins=3,
		 #                   sink_points=sink_points_mce),
		 # 'HKdr':wire_info('copper', length_hk, gauge=36, Npins=4,
		 #                  sink_points=sink_points_hk),
		 # 'HWPmotor':wire_info('pbronze', length_hk, gauge=32, Rtarget=3.5,
		 #                      Npins=8, sink_points=sink_points_hk, current=0.8/4),
		 'HWPmotor':wire_info('pbronze', length_hk, gauge=32, Rtarget=2.8,
							  Npins=8, sink_points=sink_points_hk, current=0.8/4),
		 'HWPmotorc':wire_info('copper', length_cop, gauge=26,
							   Npins=8, sink_points=sink_points_cop, current=0.8/4),
		 'HWPled':wire_info('pbronze', length_hk, gauge=36, Rtarget=20,
							Npins=8, sink_points=sink_points_hk, current=0.04 / np.sqrt(2)),
		 'HWPpd':wire_info('pbronze',length_hk, gauge=36, Rtarget=20,
						   Npins=8, sink_points=sink_points_hk, current=3e-6/np.sqrt(2)),
		 'HWPsh':wire_info('aluminum', length_hk, gauge=46, Npins=2,
						  sink_points=sink_points_hk),
}

def mul_dict(d,n):
	out = dict()
	for k, v in d.items():
		out[k] = n * v
	return out

def wiring_load(t_sft=1.5, t_mt=4.2, t_vcs1=30, t_vcs2=150, t_vv=300,
				num_inserts=6, verbose=False):

	tdict = {'sft': t_sft,
			 'mt': t_mt,
			 'mt1': t_mt,
			 'vcs1': t_vcs1,
			 'vcs2': t_vcs2,
			 'vv': t_vv}

	# total loading from cabling, mW
	keys = sorted(Winfo.keys())
	Pvec = [Winfo[key]['P'](tdict) for key in keys]
	Pveci = [Winfo[key]['PI'](tdict) for key in keys]

	Ptot_ins = dict()
	Ptoti_ins = dict()
	for p, pi in zip(Pvec, Pveci):
		for k, ki in zip(p, pi):
			if k not in Ptot_ins:
				Ptot_ins[k] = 0
				Ptoti_ins[k] = 0
			Ptot_ins[k] += p[k]
			Ptoti_ins[k] += pi[k]

	Ptot = mul_dict(Ptot_ins, num_inserts)
	Ptoti = mul_dict(Ptoti_ins, num_inserts)

	H = 2590.6 # LHe enthalpy of vaporization, J/L
	T = 86400  # seconds/day
	Lday = mul_dict(Ptot_ins, T/H) # L/day lost per insert
	Lins = mul_dict(Lday, 16) # L/flight lost per insert
	Ltot = mul_dict(Lins, num_inserts) # L lost total
	Ldayi = mul_dict(Ptoti_ins, T/H) # L/day lost per insert
	Linsi = mul_dict(Ldayi, 16) # L/flight lost per insert
	Ltoti = mul_dict(Linsi, num_inserts) # L lost total

	def print_dict(d, keys):
		return '\n'+'\n'.join('  {:4s}:  {}'.format(k, dd) for k,dd in zip(keys,d))

	if verbose:
		print('Components:',keys)
		print('Number of wires per lead:',[Winfo[k]['N'] for k in keys])
		print('Number of leads:',[Winfo[k]['Npins'] for k in keys])
		print('Total conductors',[Winfo[k]['N']*Winfo[k]['Npins'] for k in keys])
		print('Wire material',[Winfo[k]['mat'] for k in keys])
		print('Wire gauge:',[Winfo[k]['awg'] for k in keys])
		print('Wire resistance',[Winfo[k]['R'] for k in keys])
		print('Loading per component (mW/Insert):',\
			print_dict([mul_dict(p,1e3) for p in Pvec], keys))
		print('Loading per comp w/ current (mW/Insert):',\
			print_dict([mul_dict(p,1e3) for p in Pveci], keys))
		print('Total loading:')
		print('    ', mul_dict(Ptot_ins,1e3),'mW/Insert,')
		print('    ', mul_dict(Ptot,1e3),'mW')
		print('Total loading w/ current:')
		print('    ', mul_dict(Ptoti_ins,1e3),'mW/Insert,')
		print('    ', mul_dict(Ptoti,1e3),'mW')
		print('Liquid loss: ')
		print('    ', Lday,'L/Insert/day,')
		print('    ', Lins,'L/Insert/flight,')
		print('    ', Ltot,'L/flight')
		print('Liquid loss w/ current: ')
		print('    ', Ldayi,'L/Insert/day,')
		print('    ', Linsi,'L/Insert/flight,')
		print('    ', Ltoti,'L/flight')
		print('\n\n')

	return Ptot

if __name__ == "__main__":

	wiring_load(t_vcs1=30, t_vv=250, verbose=True)
