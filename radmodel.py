import numpy as np
import os

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    import code,sys
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    # print "# Interactive console. Use quit() to exit."
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        console = code.InteractiveConsole(namespace)
        if 'PYTHONSTARTUP' in os.environ:
            console.push("import os")
            console.push("execfile(os.getenv(\"PYTHONSTARTUP\"))")
        console.interact(banner=banner)
        # code.interact(banner=banner, local=namespace)
    except SystemExit:
        return

def threshold(data,low=None,high=None):
    if low is not None:
        data = np.where(data>low,data,low*np.ones_like(data))
    if high is not None:
        data = np.where(data<high,data,high*np.ones_like(data))
    return data

def parg(v,lim):
    vc = v.copy()
    vc[np.logical_not(np.isfinite(vc))] = lim
    vc[vc<lim] = lim
    return vc

def pnorm(v):
    return v/v.max()

def isarr(v):
    if isinstance(v, np.ndarray):
        return True
    return False

def uprint(v, unit='W', format='%12.6f', cd=False):
    if not v: return '%s  %s' % (format, unit) % v
    udict = {-24  : 'y',  # yocto
              -21 : 'z', # zepto
              -18 : 'a', # atto
              -15 : 'f', # femto
              -12 : 'p', # pico
              -9  : 'n', # nano
              -6  : 'u', # micro
              -3  : 'm', # mili
              0   : ' ', # --
              3   : 'k', # kilo
              6   : 'M', # mega
              9   : 'G', # giga
              12  : 'T', # tera
              15  : 'P', # peta
              18  : 'E', # exa
              21  : 'Z', # zetta
              24  : 'Y', # yotta
              }
    if cd:
        udict.update({
                -2 : 'c',   # centi
                 -1 : 'd',   # deci
                 })
    
    vlist = np.array(sorted(udict.keys()))
    lv = np.log10(np.abs(v))
    if (lv<vlist.min()): idx = 0
    elif (lv>vlist.max()): idx = -1
    else:
        idx = ((lv-vlist)<0).argmax()-1
    scale = np.power(10.,-vlist[idx])
    prefix = udict[vlist[idx]]
    return '%s %s%s' % (format, prefix, unit) % (v*scale)

def int_tabulated(y, x=None, p=5, n=None):
    # similar to IDL int_tabulate...
    if x is None: x = np.arange(len(y))
    from scipy.interpolate import splrep,splev
    if n is None: n = len(x)
    while n % (p-1) != 0: n += 1
    h = float(x.max()-x.min())/float(n)
    ix = np.arange(n+1)*h + x.min()
    iy = splev(ix,splrep(x,y,s=0),der=0)
    from scipy.integrate import newton_cotes as nc_coeff
    weights = nc_coeff(np.arange(p))[0]
    wt = np.tile(weights,[n/(p-1),1])
    yt = np.append(iy[:-1].reshape(-1,p-1),
                    iy[(p-1)::(p-1)][:,None],axis=1)
    return h * (wt * yt).sum()

def integrate(y, x=None, idx=None):
    if np.isscalar(y):
        if not isarr(x): return 0.0
        y = y*np.ones_like(x)
    if x is None: x = np.arange(len(y))
    if idx is not None:
        y = y[idx]
        x = x[idx]
    return np.trapz(y,x=x)
    # n = len(x)
    # while n % 4 != 0: n += 1
    # ix = np.linspace(x.min(),x.max(),n)
    # from scipy.interpolate import splrep,splev
    # iy = splev(ix,splrep(x,y,s=0),der=0)
    # return np.trapz(iy,x=ix)

def blackbody(f,t):
    """
    blackbody in units of W/m^2/str/Hz given freq in GHz and temp in K
    """
    from scipy.constants import h, c, k
    nu = f*1e9
    x = h*nu / (k*t)
    # return 2.*h*(nu**3)/(c**2)/(np.exp(x)-1.)
    return 2.*(k*t)*((nu/c)**2)*x/(np.exp(x)-1.)

class FilterModel(object):
    def __init__(self,name, filename=None, nfilt=1, fcent=None, width=None,
                 amp=None, wavelength=None, t_min=None, t_max=None,
                 norm=False, type='reflector', abs_filename=None,
                 thickness=None, a_min=None, a_max=None,
                 t_lowf=None, t_highf=None, a_lowf=None, a_highf=None):
        self.name = name
        self.wavelength = None
        self.trans = None
        self.type = type
        if filename is None:
            self._load_from_params(fcent=fcent, width=width, amp=amp,
                                   nfilt=nfilt, norm=norm)
        else:
            self._load_from_file(filename, nfilt=nfilt, norm=norm)
        if t_lowf is not None:
            self.trans_raw[0] = t_lowf
        if t_highf is not None:
            self.trans_raw[-1] = t_highf
        self.trans = self._interpt(wavelength=wavelength,
                                   t_min=t_min, t_max=t_max) 
        if type == 'partial':
            if abs_filename is None:
                raise ValueError,'Need filename for absorption data'
            self._load_abs_from_file(abs_filename, thickness=thickness,
                                     norm=norm)
            if a_lowf is not None:
                self.abs_raw[0] = a_lowf
            if a_highf is not None:
                self.abs_raw[-1] = a_highf
            self.abs = self._interpa(wavelength=wavelength,
                                     a_min=a_min, a_max=a_max)
            
            # correct transmission if absorption is high
            self.trans = np.where(self.trans + self.abs > 1,
                                  1 - self.abs, self.trans)
    
    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.name)
    
    def _load_from_file(self,filename,nfilt=1,norm=False):
        """
        Read in filter transmission spectrum
        """
        filename_orig = filename
        if not os.path.isfile(filename):
            realdir = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(realdir, 'rad_data', filename)
        if not os.path.isfile(filename):
            raise OSError,'Cannot find filter file %s' % filename_orig
        
        self.filename = filename
        self.nfilt = nfilt
        
        f,t = np.loadtxt(filename,unpack=True,skiprows=1)
        t = t**nfilt
        l = 1.0e4/f # microns
        l = np.append(np.insert(l,0,1e6),1e-6)
        t = np.append(np.insert(t,0,1.0),0.0)
        self.wavelength_raw = l
        if norm: t /= np.max(t)
        self.trans_raw = threshold(t,low=0.0,high=1.0)
        
    def _load_abs_from_file(self,abs_filename, thickness=2.18, norm=False):
        abs_filename_orig = abs_filename
        if not os.path.isfile(abs_filename):
            realdir = os.path.dirname(os.path.realpath(__file__))
            abs_filename = os.path.join(realdir, 'rad_data', abs_filename)
        if not os.path.isfile(abs_filename):
            raise OSError,\
                'Cannot find filter file %s' % abs_filename_orig
            
        f,a = np.loadtxt(abs_filename,unpack=True,skiprows=1)
        l = 1.0e4/f # microns
        l = np.append(np.insert(l,0,1e6),1e-6)
        # NB: thickness in mm
        a = np.append(np.insert(1-np.exp(-a*thickness),0,0.0),1.0)
        self.abs_wavelength_raw = l
        if norm: a /= np.max(a)
        self.abs_raw = a
        
    def _load_from_params(self,fcent=None, width=None, amp=None, 
                          nfilt=1,norm=False):
        """
        Generate filter transmission from parameters
        Inputs:
        freq = center frequency of filter
        width = width of filter band
        amp = amplitude at fcent
        """
        if fcent is None:
            raise ValueError, 'missing filter frequency scale'
        if width is None:
            raise ValueError, 'missing filter width'
        if amp is None:
            raise ValueError, 'missing filter transmission amplitude'
        
        self.fcent = fcent
        self.width = width
        self.amp = amp
        self.nfilt = nfilt
        
        f = np.linspace(1,50,1024)
        t = np.ones_like(f)
        p = np.where(f>=fcent)
        arg = 0.5*((f[p]-fcent)/width)**2
        t[p] = threshold(np.exp(-arg),low=1e-3)
        t = (t*amp)**nfilt
        l = 1e4/f # microns
        l = np.append(np.insert(l,0,1e6),1e-6)
        t = np.append(np.insert(t,0,1.0),0.0)
        if norm: t /= np.max(t)
        self.wavelength_raw = l
        self.trans_raw = t
        
    def _interpt(self,wavelength=None,t_min=None,t_max=None):
        if wavelength is not None:
            idx = self.wavelength_raw.argsort()
            t = np.interp(wavelength,self.wavelength_raw[idx],
                          self.trans_raw[idx])
        else: t = self.trans_raw
        t = threshold(t,low=t_min,high=t_max)
        return t

    def _interpa(self,wavelength=None,a_min=None,a_max=None):
        if wavelength is not None:
            idx = self.abs_wavelength_raw.argsort()
            a = np.interp(wavelength,self.abs_wavelength_raw[idx],
                          self.abs_raw[idx])
        else: a = self.abs_raw
        a = threshold(a,low=a_min,high=a_max)
        return a
    
    def get_trans(self,wavelength=None,t_min=0,t_max=1):
        if wavelength is not None:
            t = self._interpt(wavelength)
        else: t = self.trans
        return threshold(t,low=t_min,high=t_max)

    def get_abs(self,wavelength=None,a_min=0,a_max=1):
        if wavelength is not None:
            a = self._interpa(wavelength)
        else: a = self.abs
        return threshold(a,low=a_min,high=a_max)

    def get_emis(self,wavelength=None,t_min=0,t_max=1):
        if self.type == 'reflector':
            # return 0.01*(1.0-self.get_trans(wavelength,t_min,t_max))
            return 0.0
        elif self.type == 'partial':
            return self.get_abs(wavelength,t_min,t_max)
        elif self.type == 'absorber':
            return 1.0 - self.get_trans(wavelength,t_min,t_max)
        else:
            raise KeyError,'unknown filter type %s' % self.type
    
    def get_ref(self,wavelength=None,t_min=0,t_max=1,
                a_min=0,a_max=1):
        if self.type == 'reflector':
            return 1.0 - self.get_trans(wavelength,t_min,t_max)
        elif self.type == 'partial':
            return 1.0 - self.get_trans(wavelength,t_min,t_max) \
                - self.get_abs(wavelength,a_min,a_max)
        elif self.type == 'absorber':
            return 0.0
        else:
            raise KeyError,'unknown filter type %s' % self.type

class PolyFilter(FilterModel):
    def __init__(self, name, filename=None, thickness=1, **kwargs):
        kwargs['type'] = 'partial'
        kwargs['abs_filename'] = 'poly_abs.txt'
        super(PolyFilter,self).__init__(name, filename,
                                        thickness=thickness, **kwargs)

class MylarFilter(FilterModel):
    def __init__(self, name, filename=None, thickness=1, **kwargs):
        kwargs['type'] = 'partial'
        kwargs['abs_filename'] = 'mylar_abs_icm.txt'
        super(MylarFilter,self).__init__(name, filename,
                                        thickness=thickness, **kwargs)

class ZitexFilter(FilterModel):
    def __init__(self, name, filename=None, thickness=1, wavelength=None,
                 a_min=None, a_max=None, norm=None, **kwargs):
        self.name = name
        self.type = 'absorber'
        self._load_abs_from_file('zitex_abs_icm.txt', thickness=thickness,
                                 norm=norm)
        self.abs = self._interpa(wavelength=wavelength, a_min=a_min,
                                 a_max=a_max)
        self.trans = 1 - self.abs
        # kwargs['type'] = 'partial'
        # kwargs['abs_filename'] = 'zitex_abs_icm.txt'
        # super(ZitexFilter,self).__init__(name, filename,
        #                                  thickness=thickness, **kwargs)

class HotPressFilter(PolyFilter):
    def __init__(self,name, filename=None, thickness=2.18, **kwargs):
        super(HotPressFilter,self).__init__(name, filename, thickness,
                                        **kwargs)

class ShaderFilter(MylarFilter):
    def __init__(self, name, filename=None, thickness=0.004, t_highf=0.5,
                 a_highf=0.5, **kwargs):
        super(ShaderFilter,self).__init__(name, filename, thickness,
                                          t_highf=t_highf, a_highf=a_highf,
                                          **kwargs)

class NylonFilter(FilterModel):
    def __init__(self,thickness, wavelength, a=None, b=None, alt=False,
                 t_min=None, t_max=None, norm=False):
        self.name = 'nylon'
        self.wavelength = None
        self.trans = None
        self.type = 'absorber'
        self._load(thickness, wavelength, a=a, b=b, alt=alt,
                   t_min=t_min, t_max=t_max, norm=norm)
    
    def _load(self,thickness, wavelength, a=None, b=None, alt=False,
              t_min=None, t_max=None, norm=False):
        if a is None and b is None:
            if alt:
                a = 7.35e-5
                b = 3.55
            else:
                a = 1.5e-4
                b = 3.3
        else:
            if a is None: raise ValueError,'missing a'
            if b is None: raise ValueError, 'missing b'
        
        self.a = a
        self.b = b
        self.thickness = thickness
        self.wavelength = wavelength
        
        dx = thickness/10.0 # cm
        f = 1.0e4/wavelength # icm
        
        alpha = a*(f**b)
        arg = threshold(alpha*dx,high=13)
        t = threshold(np.exp(-arg),low=t_min,high=t_max)
        if norm: t /= np.max(t)
        self.trans = t

class Cirlex(FilterModel):
    
    def __init__(self,frequency):
        self.name = 'cirlex'
        self.frequency = frequency
        self.type = 'absorber'
    
    def get_trans(self,wavelength=None,t_min=None,t_max=None):
        if not hasattr(self,'trans'):
            # from Judy Lau's thesis, calculate the loss tangent at room 
            # temperature
            tandelta_cirlex_warm = \
                (2.0 * (0.037*(self.frequency/150)**0.52)) / 3.37
            # shift it to cold values, based on the scaling they measured
            # at 90 GHz
            tandelta_cirlex_cold = tandelta_cirlex_warm * (0.002/0.017)
            # convert that to alpha
            alpha_cirlex_cold = tandelta_cirlex_cold * \
                ((2*np.pi*np.sqrt(3.37)*self.frequency*1.0e9)/(3.0e8))
            # convert alpha to loss in percent
            trans_cirlex_cold = \
                np.exp(-alpha_cirlex_cold * (0.010 / 39.3701))
            # # convert that to alpha
            # alpha_cirlex_warm = tandelta_cirlex_warm * \
            #     ((2*np.pi*np.sqrt(3.37)*self.frequency*1.0e9)/(3.0e8))
            # # convert alpha to loss in percent
            # trans_cirlex_warm = \
            #     np.exp(-alpha_cirlex_warm * (0.010 / 39.3701))
            self.trans = threshold(trans_cirlex_cold,low=t_min,high=t_max)
        return self.trans

class Quartz(Cirlex):
    
    def __init__(self,*args,**kwargs):
        super(Quartz,self).__init__(*args,**kwargs)
        self.name = 'quartz'
    
    def get_trans(self,wavelength=None,t_min=None,t_max=None):
        if not hasattr(self,'trans'):
            trans = np.power(
                super(Quartz,self).get_trans(wavelength,t_min,t_max),0.1)
            self.trans = threshold(trans,low=t_min,high=t_max)
        return self.trans

###########################################
# New approach for easier debugging
# Treat each surface (filter, temperature stage, etc) as a radiative thing
# Calculate incident power from any one to any other
# Catalogue transmitted/reflected/absorbed terms independently
###########################################

class RadiativeSurface(object):
    
    def __init__(self, name, temperature=None, frequency=None,
                 bb=0, trans=1.0, abs=0.0, ref=0.0, incident=None,
                 wavelength=None, aperture=None, area=None,
                 antenna=False, band=None, verbose=False, **kwargs):
        self.verbose = verbose
        if self.verbose:
            print 'Initializing surface',name
        self.name = name
        self.temperature = temperature
        self.frequency = frequency
        self.wavelength = wavelength
        self.antenna = antenna
        self.band = band
        if frequency is not None and temperature is not None:
            self.bb = blackbody(frequency, temperature)
        else:
            self.bb = bb
        self.trans = trans
        self.abs = abs
        self.ref = ref
        if aperture:
            self.area = np.pi*(aperture/2.)**2
        else:
            self.area = area
        self.incident = incident
        if incident is not None: self.propagate(incident)
    
    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.name)
    
    def propagate(self, incident=None, force=False):
        if self.checkinc(incident, force=force): return
        
        if self.incident:
            self.verbose = self.incident.verbose
            
        if self.verbose:
            print 'Propagating to', self.name
            
        # get spectra
        tloc = self.get_trans()
        eloc = self.get_abs()
        rloc = self.get_ref()
        bb = self.bb
        # initialize
        self.itrans_list = []
        self.itrans = 0.0
        self.iabs_list = []
        self.iabs = 0.0
        self.iref_list = []
        self.iref = 0.0
        if self.incident:
            
            # inherit some common stuff from the incident source
            self.wavelength = self.incident.wavelength
            self.frequency = self.incident.frequency
            if self.temperature is not None and self.frequency is not None:
                self.bb = blackbody(self.frequency, self.temperature)
            self.area = self.incident.area
            
            # loop over all incident power sources
            ilist = self.incident.get_itrans_list()
            if len(ilist):
                # transmitted through this surface
                self.itrans_list = [('%s trans %s' % (iname, self.name),
                                     ii*tloc) for iname, ii in ilist]
                # absorbed by this surface
                if not np.all(eloc==0):
                    self.iabs_list = [('%s abs %s' % (iname, self.name),
                                       ii*eloc) for iname, ii in ilist]
                # reflected off this surface
                if not np.all(rloc==0):
                    self.iref_list = [('%s ref %s' % (iname, self.name),
                                       ii*rloc) for iname, ii in ilist]
        # add reemitted power to transmitted sources
        if not ( np.all(bb==0) or np.all(eloc==0) ):
            self.itrans_list.append(('%s emis' % self.name, bb*eloc))
        # total up
        if len(self.itrans_list):
            self.itrans = np.sum([x[1] for x in self.itrans_list], axis=0)
        else: self.itrans = 0.0
        if len(self.iabs_list):
            self.iabs = np.sum([x[1] for x in self.iabs_list], axis=0)
        else: self.iabs = 0.0
        if len(self.iref_list):
            self.iref = np.sum([x[1] for x in self.iref_list], axis=0)
        else: self.iref = 0.0
        
    def results(self, filename=None, mode='w', display=True, summary=False):
        if isarr(self.frequency):
            freq = self.frequency*1e9 # hz
            if self.antenna:
                from scipy.constants import c
                conv = np.power(c/freq,2)
            else: conv = self.area*np.pi # hemispherical scattering!
            self.itrans_int = integrate(conv*self.itrans, freq, idx=self.band)
            self.iabs_int = integrate(conv*self.iabs, freq, idx=self.band)
            self.iref_int = integrate(conv*self.iref, freq, idx=self.band)
            self.itrans_list_int = []
            self.iabs_list_int = []
            self.iref_list_int = []
            if len(self.itrans_list):
                self.itrans_list_int = [(x[0],
                                         integrate(conv*x[1], freq, idx=self.band))
                                        for x in self.itrans_list]
            else: self.itrans_list_int = []
            if len(self.iabs_list):
                self.iabs_list_int = [(x[0],
                                       integrate(conv*x[1], freq, idx=self.band))
                                      for x in self.iabs_list]
            else: self.iabs_list_int = []
            if len(self.iref_list):
                self.iref_list_int = [(x[0],
                                       integrate(conv*x[1], freq, idx=self.band))
                                      for x in self.iref_list]
            else: self.iref_list_int = []
        
        if not display: return
        
        if filename is None:
            import sys
            f = sys.stdout
        else:
            f = open(filename, mode)
        
        f.write('*'*80+'\n')
        f.write('%-8s: %s\n' % ('Surface', self.name))
        if self.incident:
            f.write('%-8s: %s\n' % ('Source', self.incident.name))
            f.write('%-12s: %s\n' % ('INCIDENT',uprint(self.incident.itrans_int)))
            norm = self.incident.itrans_int
        else: norm = self.itrans_int
        f.write('%-12s: %s %10.3f%%\n' % 
                ('TRANSMITTED',uprint(self.itrans_int), self.itrans_int/norm*100))
        if not summary:
            if self.itrans_int:
                for x in self.itrans_list_int:
                    f.write('  %-15s %s %10.3f%%\n' % 
                            (x[0].split('emis')[0].strip(), uprint(x[1]),
                             x[1]/self.itrans_int*100))
        f.write('%-12s: %s %10.3f%%\n' % 
                ('ABSORBED',uprint(self.iabs_int), self.iabs_int/norm*100))
        if not summary:
            if self.iabs_int:
                for x in self.iabs_list_int:
                    tag,rem = x[0].split('emis')
                    tag = tag.strip()
                    if hasattr(self,'surfaces'):
                        tag2 = 'to  %-15s' % rem.split('abs')[-1].strip()
                    else: tag2 = ''
                    f.write('  %-15s%s %s %10.3f%%\n' %
                            (tag, tag2, uprint(x[1]),
                             x[1]/self.iabs_int*100))
        f.write('%-12s: %s %10.3f%%\n' % 
                ('REFLECTED',uprint(self.iref_int), self.iref_int/norm*100))
        if not summary:
            if self.iref_int:
                for x in self.iref_list_int:
                    tag,rem = x[0].split('emis')
                    tag = tag.strip()
                    if hasattr(self,'surfaces'):
                        tag2 = 'to  %-15s' % rem.split('ref')[-1].strip()
                    else: tag2 = ''
                    f.write('  %-15s%s %s %10.3f%%\n' %
                            (tag, tag2, uprint(x[1]),
                             x[1]/self.iref_int*100))
        
        if filename is not None:
            f.close()
        
    def checkinc(self, incident, force=False):
        if incident is None:
            if self.incident is None:
                raise ValueError,'missing incident stage! %s' % self.name
            incident = self.incident
        if incident not in [None, False] and \
                not isinstance(incident, RadiativeSurface):
            raise TypeError,'incident must be an instance of RadiativeSurface'
        if isinstance(self.incident,RadiativeSurface) and \
                np.all(incident.get_itrans() == self.incident.get_itrans()) \
                and hasattr(self,'itrans'):
            # print 'Stage %s: incident surface %s has already be dealt with!' \
            #     % (self.name, incident.name)
            if force:
                return False
            return True
        self.incident = incident
        return False
        
    def checkprop(self, attr=None, incident=None):
        if attr is None or not hasattr(self,attr):
            self.propagate(incident)
            
    def get_trans(self):
        """Transmission spectrum"""
        return self.trans
    
    def get_abs(self):
        """Emission spectrum"""
        return self.abs
    
    def get_ref(self):
        """Reflection spectrum"""
        return self.ref
    
    def get_itrans(self, incident=None):
        """Transmitted power, given incident stage"""
        self.checkprop('itrans', incident)
        return self.itrans
    
    def get_itrans_list(self, incident=None):
        """Transmitted power broken down into components"""
        self.checkprop('itrans_list', incident)
        return self.itrans_list
    
    def get_iabs(self, incident=None):
        """Absorbed power, given incident stage"""
        self.checkprop('iabs', incident)
        return self.iabs
    
    def get_iabs_list(self, incident=None):
        """Absorbed power broken down into components"""
        self.checkprop('iabs_list', incident)
        return self.iabs_list
    
    def get_iref(self, incident=None):
        """Reflected power, given incident stage"""
        self.checkprop('iref', incident)
        return self.iref
    
    def get_iref_list(self, incident=None):
        """Reflected power broken down into components"""
        self.checkprop('iref_list', incident)
        return self.iref_list
    
    def get_norm_spec(self,attr):
        if not hasattr(self,attr): return None, 0
        spec = getattr(self,attr)
        if np.isscalar(spec): return None, 0
        if not isinstance(spec,np.ndarray): return None, 0
        return spec, spec.max()
    
    def _plot(self, spectra, x=None, prefix='', suffix='',
              xlim=None, ylim=None, xscale='log', yscale='log',
              **kwargs):
        
        if self.verbose:
            print 'Plotting', self.name, suffix.replace('_','')
        
        import pylab
        fig = kwargs.pop('fig', pylab.figure())
        ax = kwargs.pop('ax', pylab.gca())
        
        line_cycle = ['-','--','-.',':']
        from matplotlib import rcParams
        nc = len(rcParams['axes.color_cycle'])
        
        line_count = 0
        
        if x is None: x = self.frequency
        for v,fmt,lab,kw in spectra:
            if isarr(v):
                if ylim is not None:
                    v = parg(v, min(ylim))
                if fmt is not None:
                    ax.plot(x,v,fmt,label=lab,**kw)
                else:
                    fmt = line_cycle[int(np.floor(line_count/nc))]
                    ax.plot(x,v,fmt,label=lab,**kw)
                    line_count += 1
        
        if not len(ax.get_lines()):
            if self.verbose:
                print 'No data!'
            return
        
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(kwargs.pop('xlabel', 'Frequency [GHz]'))
        ax.set_ylabel(kwargs.pop('ylabel', 'Intensity [a.u.]'))
        ax.legend(loc=kwargs.pop('legend_loc', 'best'),
                  ncol=kwargs.pop('legend_ncol', 1))
        ax.set_title(kwargs.pop('title', self.name))
        
        filename = '%s%s%s' % (prefix, self.name, suffix)
        filename = filename.lower().replace(' ','_')
        fig.savefig(kwargs.pop('filename', filename), bbox_inches='tight')
        
        if kwargs.pop('close', True):
            pylab.close()
    
    def plot_tra(self, **kwargs):
        
        trans = self.get_trans()
        abs = self.get_abs()
        ref = self.get_ref()
        
        spectra = [
            (trans, '-b', r'$t_{\nu}$', {}),
            (ref,   '-g', r'$r_{\nu}$', {}),
            (abs,   '-r', r'$a_{\nu}$', {})
            ]
        self._plot(spectra, suffix='_coeff', ylabel='Coefficient', **kwargs)
    
    def plot_spect(self, **kwargs):
        
        if self.incident:
            iitrans, iitnorm = self.incident.get_norm_spec('itrans')
            iibb, iibnorm = self.incident.get_norm_spec('bb')
        else: iitnorm = 0.0; iibnorm = 0.0
        itrans, itnorm = self.get_norm_spec('itrans')
        ibb, ibnorm = self.get_norm_spec('bb')
        if ibnorm and itnorm and (itrans==ibb).all(): ibb = None; ibnorm = 0
        iabs, ianorm = self.get_norm_spec('iabs')
        iref, irnorm = self.get_norm_spec('iref')
        
        norm = max([iitnorm, itnorm])
        if not norm: norm = np.max([iibnorm, ibnorm, ianorm, irnorm])
        
        spectra = []
        if self.incident:
            iname = self.incident.name.replace(' ',',').upper()
            spectra += [
                (iitrans/norm, '-b', r'$I_{\nu}^{%s}$' % iname, {}),
                ]
            if iibnorm:
                spectra += [
                    (iibb/norm,   '--b', r'$B_{\nu}^{%s}$' % iname, {}),
                    ]
        spectra += [
            (itrans/norm,  '-g', r'$I_{\nu}^{T}$', {}),
            ]
        if ibb is not None:
            name = self.name.replace(' ',',').upper()
            spectra += [
                (ibb/norm,    '--g', r'$B_{\nu}^{%s}$' % name, {}),
                ]
        if iabs is not None:
            spectra += [
                (iabs/norm,    '-r', r'$I_{\nu}^{A}$', {}),
                ]
        if iref is not None:
            spectra += [
                (iref/norm,    '-c', r'$I_{\nu}^{R}$', {})
                ]
        self._plot(spectra, suffix='_spectra', **kwargs)
        
    def plot_trans(self, **kwargs):
        
        if self.incident:
            iitrans, iitnorm = self.incident.get_norm_spec('itrans')
        else: iitnorm = 0.0
        itrans, itnorm = self.get_norm_spec('itrans')
        norm = max([iitnorm, itnorm])
        if not norm: return
        
        spectra = []
        if self.incident:
            spectra += [
                (iitrans/norm, '--k', 'Incident from %s' % self.incident.name,
                 {'lw':2}),
                ]
        spectra += [
            (itrans/norm,  '-k', 'Total Transmitted', {'lw':2})
            ]
        for item in self.itrans_list:
            k,v = item
            stage = k.split()[0]
            if stage == self.name.split()[0]:
                name = k.split('emis')[0].strip()
            else:
                name = k.split()[0]
            names = [x[2] for x in spectra]
            if name in names:
                idx = names.index(name)
                tup = spectra[idx]
                spectra[idx] = (tup[0] + v/norm,) + tup[1:]
            else:
                spectra += [(v/norm, None, name, {})]
        self._plot(spectra, suffix='_trans',
                   ylabel='Transmitted Intensity [a.u.]',
                   **kwargs)
        
    def plot_abs(self, **kwargs):
        
        iabs, norm = self.get_norm_spec('iabs')
        if not norm: return
        
        spectra = []
        if self.incident:
            iitrans, norm = self.incident.get_norm_spec('itrans')
            spectra += [
                (iitrans/norm, '--k', 'Incident from %s' % self.incident.name,
                 {'lw':2}),
                ]
        
        spectra += [
            (iabs/norm,  '-k', 'Total Absorbed', {'lw':2}),
            ]
        for item in self.iabs_list:
            k,v = item
            stage = k.split()[0]
            # stage = k.split('abs')[-1].strip().split()[0]
            tag,rem = k.split('emis')
            tag = tag.strip()
            if stage != self.name.split()[0]:
                tag= tag.split()[0]
            if hasattr(self,'surfaces'):
                tag = 'On ' + rem.split('abs')[-1].strip()
            tags = [x[2] for x in spectra]
            if tag in tags:
                idx = tags.index(tag)
                tup = spectra[idx]
                spectra[idx] = (tup[0] + v/norm,) + tup[1:]
            else:
                spectra += [(v/norm, None, tag, {})]
        self._plot(spectra, suffix='_abs', ylabel='Absorbed Intensity [a.u.]',
                   **kwargs)
        
    def plot_ref(self, **kwargs):
        
        iref, norm = self.get_norm_spec('iref')
        if not norm: return
        
        spectra = []
        if self.incident:
            iitrans, norm = self.incident.get_norm_spec('itrans')
            spectra += [
                (iitrans/norm, '--k', 'Incident from %s' % self.incident.name,
                 {'lw':2}),
                ]
        
        spectra += [
            (iref/norm,  '-k', 'Total Reflected', {'lw':2}),
            ]
        for item in self.iref_list:
            k,v = item
            stage = k.split()[0]
            tag,rem = k.split('emis')
            tag = tag.strip()
            if stage != self.name.split()[0]:
                tag= tag.split()[0]
            if hasattr(self,'surfaces'):
                tag = 'Off ' + rem.split('ref')[-1].strip()
            tags = [x[2] for x in spectra]
            if tag in tags:
                idx = tags.index(tag)
                tup = spectra[idx]
                spectra[idx] = (tup[0] + v/norm,) + tup[1:]
            else:
                spectra += [(v/norm, None, tag, {})]
        self._plot(spectra, suffix='_ref', ylabel='Reflected Intensity [a.u.]',
                   **kwargs)
        
class FilterSurface(RadiativeSurface):
    def __init__(self, name, filt, **kwargs):
        super(FilterSurface,self).__init__(name, **kwargs)
        self.filter = filt
        self.trans = filt.get_trans()
        self.abs = filt.get_emis()
        self.ref = filt.get_ref()

class RadiativeStack(RadiativeSurface):
    
    def __init__(self, name, surfaces, incident=None, **kwargs):
        super(RadiativeStack, self).__init__(name, **kwargs)
        self.surfaces = surfaces
        self.incident = incident
        if incident is not None: self.propagate(incident)
    
    def propagate(self, incident=None, force=False):
        if self.checkinc(incident, force=force): return
        self.verbose = self.incident.verbose
        
        if self.verbose:
            print 'Propagating to', self.name
        
        # inherit some common stuff from the incident source
        self.wavelength = self.incident.wavelength
        self.frequency = self.incident.frequency
        if self.temperature is not None and self.frequency is not None:
            self.bb = blackbody(self.frequency, self.temperature)
        self.area = self.incident.area
        
        curinc = incident
        self.iabs_list = []
        self.iref_list = []
        self.trans = 1.0
        self.abs = 0.0
        self.ref = 0.0
        for S in self.surfaces:
            if S == self.incident:
                curinc = S
                continue
            # propagate incident power to next surface
            S.propagate(curinc, force=force)
            # get spectra
            tcur = S.get_trans()
            acur = S.get_abs()
            rcur = S.get_ref()
            # update absorbed power sourcs
            self.iabs_list.extend(S.get_iabs_list())
            # update reflected power sources. NB: totally reflected power is
            # transmitted twice through previous surfaces
            # rlist = [(rname, rr*self.trans) for rname,rr in S.get_iref_list()]
            rlist = S.get_iref_list()
            self.iref_list.extend(rlist)
            # self.ref += rcur * self.trans * self.trans
            self.ref += rcur * self.trans
            # abssivity
            self.abs = self.abs * tcur + acur
            # transmission
            self.trans *= tcur
            # proceed to next surface
            curinc = S
        # update transmitted power sources
        # NB: these have been propagated through all surfaces
        self.itrans_list = curinc.get_itrans_list()
        self.bb = curinc.bb
        # total up
        if len(self.itrans_list):
            self.itrans = np.sum([x[1] for x in self.itrans_list], axis=0)
        else: self.itrans = 0
        if len(self.iabs_list):
            self.iabs = np.sum([x[1] for x in self.iabs_list], axis=0)
        else: self.iabs = 0
        if len(self.iref_list):
            self.iref = np.sum([x[1] for x in self.iref_list], axis=0)
        else: self.iref = 0
    
    def checkinc(self, incident, force=False):
        if incident is None:
            if self.incident is None:
                if len(self.surfaces):
                    self.incident = self.surfaces[0]
        return super(RadiativeStack, self).checkinc(incident, force=force)
        
    def get_trans(self):
        """Transmission spectrum"""
        self.checkprop()
        return self.trans
    
    def get_abs(self):
        """Emission spectrum"""
        self.checkprop()
        return self.abs
    
    def get_ref(self):
        """Reflection spectrum"""
        self.checkprop()
        return self.ref
    
    def plot_tra(self, **kwargs):
        for S in self.surfaces:
            S.plot_tra(**kwargs)
        return super(RadiativeStack, self).plot_tra(**kwargs)
    
    def plot_spect(self, **kwargs):
        for S in self.surfaces:
            S.plot_spect(**kwargs)
        return super(RadiativeStack, self).plot_spect(**kwargs)
    
    def plot_trans(self, **kwargs):
        for S in self.surfaces:
            S.plot_trans(**kwargs)
        return super(RadiativeStack, self).plot_trans(**kwargs)
    
    def plot_abs(self, **kwargs):
        for S in self.surfaces:
            S.plot_abs(**kwargs)
        return super(RadiativeStack, self).plot_abs(**kwargs)
    
    def plot_ref(self, **kwargs):
        for S in self.surfaces:
            S.plot_ref(**kwargs)
        return super(RadiativeStack, self).plot_ref(**kwargs)
    
    def results(self, display_this=True, summary=False, **kwargs):
        if not summary or (summary and not display_this):
            for S in self.surfaces:
                S.results(summary=summary, **kwargs)
        if display_this:
            return super(RadiativeStack, self).results(summary=summary, **kwargs)
    
###########################################

class RadiativeModel(object):
    
    # frequency above which the sky is a simple 273K blackbody
    _MAX_FATMOS = 400.0
    # emissivity assumed for 273K atmosphere beyond 1 THz
    _ATMOS_EMIS = 1e-1
    
    def __init__(self,**kwargs):
        self.params = dict()
        self._initialized = False
        profile = kwargs.pop('profile',None)
        self.verbose = kwargs.pop('verbose', False)
        self.set_defaults(**kwargs)
        if profile is not None:
            self.load_profile(profile)
        self.set_band(**kwargs)
        # self.pretty_print_params()
        self._reload()
        
    def set_defaults(self,**kwargs):
        # temperatures
        self.params['tsky'] = kwargs.pop('tsky',2.73)
        self.params['esky'] = kwargs.pop('esky',1.0)
        self.params['tvcs1'] = kwargs.pop('tvcs1',35.0)
        self.params['tvcs2'] = kwargs.pop('tvcs2',130.0)
        self.params['t4k'] = kwargs.pop('t4k',5.0)
        self.params['atmos'] = kwargs.pop('atmos',True)
        
        # detector quantum efficiency
        self.params['eta'] = kwargs.pop('eta',0.4)
        self.params['esubk'] = kwargs.pop('esubk',0.25) # subk emissivity
        
        self.params['tsubk'] = kwargs.pop('tsubk',0.3)
        
        self.params['t_hp_min'] = kwargs.pop('t_hp_min',1e-5)
        self.params['t_sh_min'] = kwargs.pop('t_sh_min',1e-5)
        self.params['a_hp_min'] = kwargs.pop('a_hp_min',0.01)
        self.params['a_sh_min'] = kwargs.pop('a_sh_min',1e-5)
        self.params['t_ny_min'] = kwargs.pop('t_ny_min',1e-8)
        
        # window properties
        self.params['window'] = kwargs.pop('window',True)
        # emissivity at center freq
        self.params['window_emis'] = kwargs.pop('window_emis',1e-3)
        self.params['window_abs_min'] = kwargs.pop('window_abs_min',1e-3)
        self.params['window_thickness'] = kwargs.pop('window_thickness',3.175)
        
        # temperature
        self.params['twin'] = kwargs.pop('window_temp',
                                         kwargs.pop('twin',273.0))
        # emissivity index
        self.params['window_beta'] = kwargs.pop('window_beta',2)
        
        # stop properties
        self.params['t2k'] = kwargs.pop('t2k',kwargs.pop('t_stop',2.0))
        self.params['spill_frac'] = kwargs.pop('spill_frac',0.1)
        
        # nylon conductivity
        # self.params['g_nylon'] = kwargs.pop('g_nylon',3e-5) # W/K
        self.params['vcs1_nylon_dt'] = kwargs.pop('vcs1_nylon_dt',30) # K
        self.params['4k_nylon_dt'] = kwargs.pop('vcs1_nylon_dt',0) # K
        
        # aperture diameter
        self.params['aperture'] = kwargs.pop('aperture',0.3) # m
        
        # directories and files
        thisdir = os.path.dirname(os.path.realpath(__file__))
        datdir = kwargs.pop('datdir', os.path.join(thisdir, 'rad_data'))
        self.params['datdir'] = datdir
        
        figdir = kwargs.pop('figdir', os.path.join(thisdir, 'figs'))
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        self.params['figdir'] = figdir
        
        atmfile = kwargs.pop('atmfile', 'amatm.dat')
        if not os.path.exists(atmfile):
            atmfile = os.path.join(datdir, atmfile)
        self.params['atmfile'] = atmfile
        spectfile = kwargs.pop('spectfile', 'spectrum_150ghz.dat')
        if not os.path.exists(spectfile):
            spectfile = os.path.join(datdir, spectfile)
        self.params['spectfile'] = spectfile
        
    def update_params(self,*args,**kwargs):
        self.params.update(*args,**kwargs)
    
    def get_param(self,param,default=None):
        if param not in self.params and default is not None:
            self.params[param] = default
        elif default is not None and self.params[param] != default:
            self.params[param] = default
        if param not in self.params: return None
        return self.params[param]
    
    def load_profile(self,profile):
        tsky,esky,tvcs2,tvcs1,t4k,atmos = np.loadtxt(profile,unpack=True)
        self.params['tsky'] = tsky[0]
        self.params['esky'] = esky[0]
        self.params['tvcs2'] = tvcs2[0]
        self.params['tvcs1'] = tvcs1[0]
        self.params['t4k'] = t4k[0]
        self.params['atmos'] = atmos[0]
        
    def print_params(self):
        for k in sorted(self.params.keys()):
            print '%s = %r' % (k,self.params[k])
    
    def pretty_print_params(self):
        def tprint(v):
            return uprint(v, unit='K', format='%8.3f')
        print '%-20s: %s' % ('Sky temp', tprint(self.params['tsky']))
        print '%-20s: %8.3f' % ('Sky emissivity', self.params['esky'])
        print '%-20s: %8.3f' % ('Atmosphere fraction', self.params['atmos'])
        print '%-20s: %s' % ('VCS2 temp', tprint(self.params['tvcs2']))
        print '%-20s: %s' % ('VCS1 temp', tprint(self.params['tvcs1']))
        print '%-20s: %s' % ('4K temp', tprint(self.params['t4k']))
        print '%-20s: %s' % ('2K temp', tprint(self.params['t2k']))
        print '%-20s: %8.3f' % ('Stop throughput', self.params['spill_frac'])
        
    def load_filters(self,norm=True):
        t_hp_min = self.params['t_hp_min']
        t_sh_min = self.params['t_sh_min']
        a_hp_min = self.params['a_hp_min']
        a_sh_min = self.params['a_sh_min']
        t_ny_min = self.params['t_ny_min']
        fopts = dict(wavelength=self.wavelength)
        self.filters = {
            'c8-c8': ShaderFilter('c8-c8','spider_filters_c8-c8.txt',
                                  t_min=t_sh_min, a_min=a_sh_min,
                                  norm=norm, **fopts),
            'c12-c16': ShaderFilter('c12-c16','spider_filters_c12-c16.txt',
                                    t_min=t_sh_min, a_min=a_sh_min,
                                    norm=norm, **fopts),
            'c15': ShaderFilter('c15','spider_filters_c15.txt',
                                t_min=t_sh_min, a_min=a_sh_min,
                                norm=norm, **fopts),
            'c16-c25': ShaderFilter('c16-c25','spider_filters_c16-c25.txt',
                                    t_min=t_sh_min, a_min=a_sh_min,
                                    norm=norm, **fopts),
            'c30': ShaderFilter('c30','spider_filters_c30.txt',
                                t_min=t_sh_min, a_min=a_sh_min,
                                norm=norm, **fopts),
            '12icm': HotPressFilter('12icm',
                                    'spider_filters_w1078_12icm.txt',
                                    t_min=t_hp_min, a_min=a_hp_min,
                                    thickness=2.18, norm=norm, **fopts),
            '7icm': HotPressFilter('7icm','spider_filters_w1522_7icm.txt',
                                   t_min=t_hp_min, a_min=a_hp_min,
                                   thickness=2.18, norm=norm, **fopts),
            '4icm': HotPressFilter('4icm','spider_filters_4icm.txt',
                                   t_min=t_hp_min, a_min=a_hp_min,
                                   thickness=2.18, norm=norm, **fopts),
            '6icm': HotPressFilter('6icm','spider_filters_6icm.txt',
                                   t_min=t_hp_min, a_min=a_hp_min,
                                   thickness=2.18, norm=norm, **fopts),
            '10icm': HotPressFilter('10icm',fcent=8.2,width=1.5,amp=0.93,
                                    t_min=t_hp_min, a_min=a_hp_min,
                                    thickness=2.18, **fopts),
            '18icm': HotPressFilter('18icm',fcent=17.0,width=2.2,amp=0.93,
                                    t_min=t_hp_min, a_min=a_hp_min,
                                    thickness=2.18, **fopts),
            'ar90ny': ZitexFilter('ar90ny',t_min=t_ny_min,
                                  thickness=0.381, **fopts),
            'ar150ny': ZitexFilter('ar150ny', t_min=t_ny_min,
                                   thickness=0.584, **fopts),
            'ar90pe': ZitexFilter('ar90pe', t_min=t_ny_min,
                                  thickness=0.406, **fopts),
            'ar150pe': ZitexFilter('ar150pe', t_min=t_ny_min,
                                   thickness=0.610, **fopts),
            'nylon': NylonFilter(2.38, t_min=t_ny_min, norm=norm,
                                 **fopts),
            'cirlex': Cirlex(frequency=self.frequency),
            'quartz': Quartz(frequency=self.frequency),
            }
        # print 'Available filters: %r' % sorted(self.filters.keys())
        return self.filters
        
    def load_atm(self,filename=None,fmax=_MAX_FATMOS,emax=_ATMOS_EMIS):
        filename = self.get_param('atmfile',filename)
        data = np.loadtxt(filename,unpack=True)
        xout = data[0]
        yout = data[5]
        if fmax is not None:
            p = np.where(xout<fmax)[0]
            f_sky = np.append(np.insert(xout[p],0,0),1e6)
            t_sky = np.append(np.insert(yout[p],0,0),0)
        else:
            f_sky = np.append(np.insert(xout,0,0),1e6)
            t_sky = np.append(np.insert(yout,0,0),0)
        f_sky = threshold(f_sky,low=1e-12)
        t_sky = threshold(t_sky,low=1e-12)
        Inu_atmos = blackbody(f_sky,t_sky)
        self.Inu_atmos = np.where(
            self.frequency>fmax,
            emax*blackbody(self.frequency,273.0),
            np.interp(self.frequency,f_sky,Inu_atmos))
        return self.Inu_atmos
    
    def load_spectrum(self,filename=None):
        filename = self.get_param('spectfile',filename)
        f,t = np.loadtxt(filename,unpack=True)
        fcent = self.get_param('fcent')
        p = np.where((f<0.8*2*fcent)*(f>fcent/3.0))[0]
        f = f[p]
        t = threshold(t[p],low=0)
        # if fcent < 148: f *= fcent/148.0
        f = np.append(np.insert(f,0,1),1e6)
        t = np.append(np.insert(t,0,0),0)
        l = 300./f*1000.0
        idx = l.argsort()
        self.spectrum = np.interp(self.wavelength,l[idx],t[idx])
        self.spectrum /= self.spectrum.max()
        return self.spectrum
        
    def set_band(self,**kwargs):
        res = self.get_param('res',kwargs.pop('res',1000))
        fcent = self.get_param('fcent',kwargs.pop('fcent',148))
        bw = self.get_param('bw',kwargs.pop('bw',0.40))
        bandlo = self.get_param('bandlo',kwargs.pop('bandlo',3))
        bandhi = self.get_param('bandhi',kwargs.pop('bandhi',1000))
        
        self.wavelength = np.logspace(4,0,res) # wavelength in um
        self.frequency = 300.0/(self.wavelength/1000.0) # frequency in GHz
        
        self.id_band = np.where((self.frequency>bandlo) * 
                                 (self.frequency<bandhi))[0]
        
        blo = np.max([fcent*(1.0-bw/2),1.0])
        bhi = fcent*(1.0+bw/2)
        self.id_band2 = np.where((self.frequency>blo) * 
                                  (self.frequency<bhi))[0]
        
        if self._initialized: self._reload()
        
    def _reload(self):
        self.load_atm()
        self.load_spectrum()
        self.load_filters()
        if not self._initialized: self._initialized = True
        
    def run(self, tag=None, plot=False, interactive=False, summary=False,
            filter_stack={'vcs2':['c8-c8','c8-c8','c8-c8','c12-c16'],
                          'vcs1':['c12-c16','c16-c25','c16-c25','12icm'],
                          '4k':['10icm','nylon'],
                          '2k':['7icm'],
                          }, **kwargs):
        
        # abscissa and conversion factors
        freq = self.frequency
        wlen = self.wavelength
        
        # aperture area
        area = np.pi*(self.params['aperture']/2.)**2
        
        # the sky
        bb_sky = blackbody(freq,self.params['tsky'])*self.params['esky']
        if self.params['atmos']:
            bb_sky += self.Inu_atmos
        Rsky = RadiativeSurface('Sky', trans=0.0, abs=1.0, bb=bb_sky,
                                wavelength=wlen, frequency=freq,
                                area=area, incident=False,
                                verbose=self.verbose)
        surfaces = [Rsky]
        
        if self.params['window']:
            # window spectrum
            # window_emis = self.params['window_emis'] * \
            #     (freq/self.params['fcent'])**self.params['window_beta']
            # window_emis = threshold(window_emis,high=1.0)
            Fwin = PolyFilter('window', fcent=freq.max(), width=0,
                              amp=1.0, wavelength=wlen,
                              thickness=self.params['window_thickness'],
                              a_min=self.params['window_abs_min'])
            window_emis = Fwin.get_emis()
            window_trans = 1 - window_emis
            bb_win = blackbody(freq, self.params['twin'])
            Rwin = RadiativeSurface('Window', trans=window_trans,
                                    abs=window_emis, bb=bb_win)
            surfaces.append(Rwin)
        
        # assemble the filter stages into RadiativeStack objects
        def make_stack(stage):
            T = self.params['t%s'%stage]
            bb = blackbody(freq,T)
            stack = [FilterSurface('%s %s %s' % (stage.upper(),
                                                 chr(ord('A')+i),f),
                                   self.filters[f], bb=bb)
                     for i,f in enumerate(filter_stack[stage])]
            
            # hot nylon
            dt = self.get_param('%s_nylon_dt' % stage)
            if dt and 'nylon' in filter_stack[stage]:
                nyidx = filter_stack[stage].index('nylon')
                nybb = blackbody(freq, T+dt)
                stack[nyidx].bb = nybb
                # check for AR coats
                if filter_stack[stage][nyidx-1].startswith('ar'):
                    stack[nyidx-1].bb = nybb
                    stack[nyidx+1].bb = nybb
                
            return RadiativeStack(stage.upper(), stack)
        
        Rvcs2 = make_stack('vcs2')
        Rvcs1 = make_stack('vcs1')
        R4k = make_stack('4k')
        R2k = make_stack('2k')
        spill = self.params['spill_frac']
        if spill:
            R2k.surfaces += [
                # hack! spillover from optics sleeve
                RadiativeSurface('2K Spillover', bb=R2k.surfaces[-1].bb,
                                 abs=spill)
                ]
        surfaces += [Rvcs2, Rvcs1, R4k, R2k]
        
        # sub-K loading (use trans=1 to pass through to detector,
        # but bb=0 to ignore loading onto detector)
        Rsubk = RadiativeSurface('sub-K', abs=self.params['esubk'])
        surfaces.append(Rsubk)
        
        # detector loading with bandpass
        eta = self.params['eta']
        t = self.spectrum # detector FTS spectrum
        Rband = RadiativeSurface('Bandpass', trans=t,
                                 antenna=True, band=self.id_band2)
        surfaces.append(Rband)
        Rdet = RadiativeSurface('Det', trans=1-eta, abs=eta,
                                antenna=True, band=self.id_band2)
        surfaces.append(Rdet)
        
        # assemble the whole stack and propagate the sky through it
        self.tag = tag
        self.stack = RadiativeStack('TOTAL', surfaces, incident=Rsky)
        
        # second pass to account for loading on nylon
        # if any(['nylon' in f for f in filter_stack.values()]):
        #     g_nylon = self.params['g_nylon']
        #     if g_nylon and np.isfinite(g_nylon):
        #         self.stack.results(display=False)
        #         for S in self.stack.surfaces:
        #             stage = S.name.lower()
        #             if stage not in filter_stack: continue
        #             if 'nylon' not in filter_stack[stage]: continue
        #             for idx,SS in enumerate(S.surfaces):
        #                 if 'nylon' in SS.name:
        #                     dT = SS.iabs_int / g_nylon
        #                     print '%-6s' % stage.upper(), \
        #                         'nylon dT = %s' % uprint(dT,'K','%8.3f')
        #                     T = self.params['t%s'%stage] + dT
        #                     bb = blackbody(freq, T)
        #                     SS.bb = bb
        #                     if S.surfaces[idx-1].name.split()[-1].startswith('ar'):
        #                         S.surfaces[idx-1].bb = bb
        #                         S.surfaces[idx+1].bb = bb
        #         self.stack.propagate(force=True)
        
        # print results
        self.results(summary=summary)
        
        # plot results
        if plot:
            self.plot(tag=tag, interactive=interactive)
        return self.stack
    
    def results(self, summary=False):
        print '*'*80
        if hasattr(self,'tag') and self.tag:
            print 'MODEL:',self.tag.replace('_',' ')
        print '*'*80
        self.pretty_print_params()
        self.stack.results(display_this=False, summary=summary)
    
    def plot(self, tag=None, interactive=False, **kwargs):
        if not interactive:
            import sys
            if 'matplotlib.backends' not in sys.modules:
                from matplotlib import use
                use('agg')
        
        figdir = self.params['figdir']
        prefix = '%s/' % figdir
        if tag:
            figdir = os.path.join(figdir,tag)
            prefix = '%s/%s_' % (figdir, tag)
        if not os.path.exists(figdir): os.mkdir(figdir)
        
        for S in self.stack.surfaces:
            if S.name == 'Det':
                pargs = dict(
                    xlim=[50,250],
                    xscale='linear',
                    )
            else:
                pargs = dict(
                    xlim=None,
                    xscale='log',
                    )
            pargs['prefix'] = prefix
            
            S.plot_tra(ylim=[1e-3,1.1], **pargs)
            S.plot_spect(ylim=[1e-8,1.1], **pargs)
            S.plot_trans(ylim=[1e-8,1.1], **pargs)
            S.plot_abs(ylim=[1e-8,1.1], **pargs)
            S.plot_ref(ylim=[1e-8,1.1], **pargs)

def main(model_class=RadiativeModel):
    models = {
        1: {
            'filter_stack': {'vcs2':['c8-c8','c8-c8','c8-c8','c12-c16'],
                             'vcs1':['c12-c16','c16-c25','c16-c25','12icm'],
                             '4k':['10icm','nylon'],
                             '2k':['7icm'],
                             },
            'tag': 'default'
            },
        2: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz'
            },
        3: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['18icm','10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_18icm'
            },
        4: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['cirlex','10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_hwp'
            },
        5: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['quartz','10icm','nylon'],
                             '2k':['4icm'],
                             },
            'tag': '90ghz_hwp',
            'opts': dict(fcent=94, spectfile='spectrum_90ghz.dat')
            },
        6: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['cirlex','10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_hwp_300k',
            'opts': dict(tsky=300, atmos=0)
            },
        7: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm'],
                             '4k':['10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_nonylon'
            },
        8: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_300k',
            'opts': dict(tsky=300, atmos=0)
            },
        9: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm'],
                             '4k':['10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_300k_nonylon',
            'opts': dict(tsky=300, atmos=0)
            },
        10: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm','nylon'],
                             '4k':['10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_nowin',
            'opts': dict(window=False)
            },
        11: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm'],
                             '4k':['10icm','nylon'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_nowin_nonylon',
            'opts': dict(window=False)
            },
        12: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm',
                                     'ar150ny','nylon','ar150ny'],
                             '4k':['ar150pe','10icm','ar150pe',
                                   'ar150ny','nylon','ar150ny'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_ar',
            'opts': dict(g_nylon=1e-4)
            },
        13: {
            'filter_stack': {'vcs2':['c15','c15','c30','c30'],
                             'vcs1':['c15','c30','c30','12icm'],
                             '4k':['ar150pe','10icm','ar150pe',
                                   'ar150ny','nylon','ar150ny'],
                             '2k':['6icm'],
                             },
            'tag': '150ghz_ar_nonylon'
            },
        }
    
    import argparse as ap
    P = ap.ArgumentParser(add_help=True)
    P.add_argument('model',nargs='?',default=1,type=int,
                   choices=models.keys(), metavar='num',
                   help='Preset model number. Choices are: %s' % 
                   ', '.join('%d (%s)' % (k,v['tag']) for k,v in models.items()))
    P.add_argument('-i','--interactive',default=False,
                   action='store_true',help='show plots')
    P.add_argument('-p','--plot',default=False,
                   action='store_true',help='make plots')
    P.add_argument('-s','--summary',default=False,
                   action='store_true',help='print short summary')
    P.add_argument('-v','--verbose', default=False,
                   action='store_true',help='verbose mode, for debugging')
    args = P.parse_args()
    
    if not args.plot: args.interactive = False
    
    opts = dict(verbose=args.verbose, fcent=148, spectfile='145GHzSpectrum.dat')
    
    if args.model not in models:
        raise ValueError,'unrecognized model number %d' % args.model
    
    model = models[args.model]
    if 'opts' in model:
        opts.update(**model['opts'])
    
    M = model_class(**opts)
    M.run(filter_stack=model['filter_stack'], tag=model['tag'],
          plot=args.plot, interactive=args.interactive,
          summary=args.summary)

if __name__ == "__main__":
    
    main(model_class=RadiativeModel)
