import numpy as np
from scipy.interpolate import interp1d

def d2awg(d):
    return 36 - np.round(((np.log(d)-np.log(0.000127))*39/np.log(92))/2)*2

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
    f = interp1d(t,k,'quadratic')
    return f(tin)

def manganin_cv(tin):
    t=[4.0,10.0,20.0,80.0,150.0,300.0]
    k=[0.5,2.0,3.3,13.0,16.0,22.0]
    f = interp1d(t,k,'quadratic')
    return f(tin)

def copper_cv(tin):
    # assumes RRR=50
    t=[4.0,10.0,20.0,80.0,150.0,300.0]
    # k=[9.0,30.0,70.0,300.0,700.0,1100.0,600.0,410.0,400.0]
    k = [318.0,778.0,1368.0,500.0,408.0,392.0]
    f = interp1d(t,k,'quadratic')

    return f(tin)

def al6061_cv(tin):
    # Al 6061-T6 thermal conductivity [W/m-K]
    a=[0.07981,1.0957,-0.07277,0.08084,0.02803,-0.09464,0.04179,-0.00571,0]
    out = np.ones(len(tin))*a[0]
    for i in range(1,len(a)):
        out += a[i] * np.log10(tin)**i
    out = 10.0**out

    # scale linearly for electrons
    Tpiv = 5
    p = np.where(tin < Tpiv)[0]
    tref = al6061_cv(Tpiv)
    out[p] = tref * (tin[p]/Tpiv)
    return out

def pbronze_r(tin,gauge=36):
    t=[4.2, 77, 305]
    r=np.array([8.56, 8.83, 10.3])*awg2a(36)/awg2a(gauge)
    f = interp1d(t,r,'quadratic')
    return f(tin)

def manganin_r(tin,gauge=36):
    t=[4.2, 77, 305]
    r=np.array([34.6, 36.5, 38.8])*awg2a(36)/awg2a(gauge)
    f = interp1d(t,r,'quadratic')
    return f(tin)
    
def copper_r(tin,gauge=30):
    t=[4.2, 77, 305]
    r=np.array([0.003, 0.04, 0.32])*awg2a(30)/awg2a(gauge)
    f = interp1d(t,r,'quadratic')
    return f(tin)

def load_func(func,hstemps,lengths):
    tmin = np.min(hstemps)
    # tin = np.linspace(np.min(hstemps),np.max(hstemps),100)
    # P = 0
    P = np.zeros(len(lengths)-1)
    for ii,length in enumerate(lengths[1:]):
        tarr = np.linspace(tmin,hstemps[ii+1],100)
        karr = func(tarr)
        P[ii] = np.trapz(karr,tarr)/(length-lengths[ii])
    return P

def wire_info(material,length,gauge=None,Rtarget=None,Npins=1,
              sink_points={'mt':0,'vv_mt':1}):
    if material is 'manganin':
        rfunc = manganin_r
        kfunc = manganin_cv
        gauge0 = 36
    elif material is 'pbronze':
        rfunc = pbronze_r
        kfunc = pbronze_cv
        gauge0 = 36
    elif material is 'copper':
        rfunc = copper_r
        kfunc = copper_cv
        gauge0 = 30
    
    Tarr = np.linspace(4.2,300,100)
    dT = Tarr[-1]-Tarr[0]
    
    if gauge is not None:
        if gauge == 0:
            gauge=gauge0;
        area = awg2a(gauge)
        aol = area/length
        R = np.trapz(rfunc(Tarr,gauge),Tarr)/dT*length # resistance, Ohms
        if Rtarget is not None:
            N = np.int(np.ceil(R/Rtarget))
        else:
            N = 1
    else:
        R0 = np.trapz(rfunc(Tarr,gauge0),Tarr)/dT*length
        area = R0/Rtarget*awg2a(gauge0)
        aol = area/length
        gauge = a2awg(area)
        R = R0*awg2a(gauge0)/awg2a(gauge)
        N = 1
        
    # H = 1000.0*2590.6 # enthalpy of vaporization, mJ/L
    # Tflight = 86400*30 # length of flight
    
    def calc_load(tdict):
        keys = sorted(sink_points.keys(), key=lambda x: sink_points[x])
        tvec = [tdict[k.split('_')[0]] for k in keys]
        lvec = [sink_points[k] for k in keys]
        pvec = load_func(kfunc,tvec,lvec)*aol*float(N)*float(Npins) # watts
        pdict = dict()
        for idx,k in enumerate(keys[1:]):
            if '_' not in k: continue
            pdict[k] = pvec[idx]
        return pdict

    info = {'mat':material,'R':R/N,'awg':gauge,'N':N,'Npins':Npins,'P':calc_load,
            'sink':sink_points}
    return info
    # L = P*Tflight/H

def shield_info(material,length,width,thickness=0.005):
    if material is 'aluminum':
        kfunc = al6061_cv
        thick0 = 0.005
    elif material is 'mylar':
        kfunc = al6061_cv
        thick0 = 0.001

# wiring parameters
length_hk = 5*12.0*0.0254 # length of cryogenically active wiring, m
sink_points_hk = {'mt': 0, 'vcs1_mt': 0.4, 'vv_vcs1': 1}

length_mce = 3 # length of cryogenically active wiring, m
sink_points_mce = {'mt': 0, 'vcs1_mt': 0.2, 'vv_vcs1': 1}

# wire material for each component
Winfo = {'HK':wire_info('manganin', length_hk, gauge=36, Npins=50*2,
                        sink_points=sink_points_hk),
         'MCE':wire_info('manganin', length_mce, gauge=36, Npins=3*100,
                         sink_points=sink_points_mce),
         'motor':wire_info('pbronze', length_hk, gauge=32, Rtarget=3.5,
                           Npins=8, sink_points=sink_points_hk),
         'led':wire_info('pbronze', length_hk, gauge=36, Rtarget=20,
                         Npins=8, sink_points=sink_points_hk),
         'pd':wire_info('pbronze',length_hk, gauge=36, Rtarget=20,
                        Npins=8, sink_points=sink_points_hk)}

def mul_dict(d,n):
    out = dict()
    for k, v in d.items():
        out[k] = n * v
    return out

def wiring_load(t_sft=1.5, t_mt=4.2, t_vcs1=30, t_vcs2=150, t_vv=300,
                num_inserts=6, verbose=False):

    tdict = {'sft': t_sft,
             'mt': t_mt,
             'vcs1': t_vcs1,
             'vcs2': t_vcs2,
             'vv': t_vv}

    # total loading from cabling, mW
    keys = Winfo.keys()
    Pvec = [Winfo[key]['P'](tdict) for key in keys]

    Ptot_ins = dict()
    for p in Pvec:
        for k in p:
            if k not in Ptot_ins:
                Ptot_ins[k] = 0
            Ptot_ins[k] += p[k]

    Ptot = mul_dict(Ptot_ins, num_inserts)

    H = 2590.6 # LHe enthalpy of vaporization, J/L
    T = 86400 # seconds/day
    Lday = mul_dict(Ptot_ins, T/H) # L/day lost per insert
    Lins = mul_dict(Lday, 16) # L/flight lost per insert
    Ltot = mul_dict(Lins, num_inserts) # L lost total

    if verbose:
        print 'Components:',keys
        print 'Number of wires per lead:',[Winfo[k]['N'] for k in keys]
        print 'Number of leads:',[Winfo[k]['Npins'] for k in keys]
        print 'Total conductors',[Winfo[k]['N']*Winfo[k]['Npins'] for k in keys]
        print 'Wire material',[Winfo[k]['mat'] for k in keys]
        print 'Wire gauge:',[Winfo[k]['awg'] for k in keys]
        print 'Wire resistance',[Winfo[k]['R'] for k in keys]
        print 'Loading per component (mW/Insert):',[mul_dict(p,1e3) for p in Pvec]
        print 'Total loading:',mul_dict(Ptot_ins,1e3),'mW/Insert,', \
            mul_dict(Ptot,1e3),'mW'
        print 'Liquid loss: ',Lday,'L/Insert/day,', \
            Lins,'L/Insert/flight,',Ltot,'L/flight'

    return Ptot

if __name__ == "__main__":

    wiring_load(t_vcs1=30, t_vv=250, verbose=True)
