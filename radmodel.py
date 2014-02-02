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

def uprint(v, unit='W', format='%10.6f', cd=False):
    if not v: return '%s %s' % (format, unit) % v
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
    lv = np.log10(v)
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
                 e_max=0.0, norm=False, type='shader', abs_filename=None,
                 thickness=None, a_min=None, a_max=None):
        self.name = name
        self.wavelength = None
        self.trans = None
        self.type = type
        if filename is None:
            self._load_from_params(fcent=fcent, width=width, amp=amp,
                                   nfilt=nfilt, norm=norm)
        else:
            filename_orig = filename
            if not os.path.isfile(filename):
                realdir = os.path.dirname(os.path.realpath(__file__))
                filename = os.path.join(realdir, 'rad_data', filename)
            if not os.path.isfile(filename):
                raise OSError,'Cannot find filter file %s' % filename_orig
            self._load_from_file(filename, nfilt=nfilt, norm=norm)
        self.emis_max = e_max
        self.trans = self._interpt(wavelength=wavelength,
                                   t_min=t_min, t_max=t_max) 
        if type == 'metalmesh':
            if abs_filename is None:
                raise ValueError,'Need filename for absorption data'
            abs_filename_orig = abs_filename
            if not os.path.isfile(abs_filename):
                realdir = os.path.dirname(os.path.realpath(__file__))
                abs_filename = os.path.join(realdir, 'rad_data', abs_filename)
            if not os.path.isfile(abs_filename):
                raise OSError,\
                    'Cannot find filter file %s' % abs_filename_orig
            self._load_abs_from_file(abs_filename, thickness=thickness,
                                     norm=norm)
            self.abs = self._interpa(wavelength=wavelength,
                                     a_min=a_min, a_max=a_max)
        
            # correct transmission if absorption is high
            self.trans = np.where(self.trans + self.abs > 1,
                                  self.trans - (self.trans + self.abs - 1),
                                  self.trans)
        
    def _load_from_file(self,filename,nfilt=1,norm=False):
        """
        Read in filter transmission spectrum
        """
        if not os.path.isfile(filename):
            raise OSError,'file %s not found' % filename
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
        
    def _load_abs_from_file(self,filename, thickness=2.18, norm=False):
        if not os.path.isfile(filename):
            raise OSError,'file %s not found' % filename
        f,a = np.loadtxt(filename,unpack=True,skiprows=1)
        l = 1.0e4/f # microns
        l = np.append(np.insert(l,0,1e6),1e-6)
        # NB: thickness in mm
        a = np.append(np.insert(1-np.exp(-a*thickness/11),0,0.0),1.0)
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
        if self.type == 'shader':
            # return 0.01*(1.0-self.get_trans(wavelength,t_min,t_max))
            return 0.0
        elif self.type == 'metalmesh':
            return self.get_abs(wavelength,t_min,t_max)
        elif self.type == 'absorber':
            return 1.0 - self.get_trans(wavelength,t_min,t_max)
        else:
            raise KeyError,'unknown filter type %s' % self.type
    
    def get_ref(self,wavelength=None,t_min=0,t_max=1,
                a_min=0,a_max=1):
        if self.type == 'shader':
            return 1.0 - self.get_trans(wavelength,t_min,t_max)
        elif self.type == 'metalmesh':
            return 1.0 - self.get_trans(wavelength,t_min,t_max) \
                - self.get_abs(wavelength,a_min,a_max)
        elif self.type == 'absorber':
            return 0.0
        else:
            raise KeyError,'unknown filter type %s' % self.type

class MetalMeshFilter(FilterModel):
    def __init__(self,name, filename=None, thickness=2.18, **kwargs):
        kwargs['type'] = 'metalmesh'
        kwargs['abs_filename'] = 'poly_abs.txt'
        super(MetalMeshFilter,self).__init__(name, filename=filename,
                                             thickness=thickness, **kwargs)

class NylonFilter(FilterModel):
    def __init__(self,thickness, wavelength, a=None, b=None, alt=False,
                 t_min=None, t_max=None, e_max=0.0, norm=False):
        self.name = 'nylon'
        self.wavelength = None
        self.trans = None
        self.type = 'absorber'
        self._load(thickness, wavelength, a=a, b=b, alt=alt,
                   t_min=t_min, t_max=t_max, norm=norm)
        self.emis_max = e_max
    
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
    
    def get_trans(self,wavelength=None,t_min=None,t_max=None):
        if not hasattr(self,'trans'):
            trans = np.power(
                super(Quartz,self).get_trans(wavelength,t_min,t_max),0.1)
            self.trans = threshold(trans,low=t_min,high=t_max)
        return self.trans

###########################################
# New approach for easier debugging
# Treat each elementer (filter, temperature stage, etc) as a radiative thing
# Calculate incident power from any one to any other
# Catalogue transmitted/reflected/absorbed terms independently
###########################################

class RadiativeElement(object):
    
    def __init__(self, name, temperature=None, frequency=None,
                 bb=0, trans=1.0, abs=0.0, ref=0.0, incident=None,
                 wavelength=None, aperture=None, area=None,
                 single_moded=False, band=None, **kwargs):
        print 'Initializing element',name
        self.name = name
        self.temperature = temperature
        self.frequency = frequency
        self.wavelength = wavelength
        self.single_moded = single_moded
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
    
    def propagate(self, incident=None, force=False):
        if self.checkinc(incident, force=force): return
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
                # transmitted through this element
                self.itrans_list = [('%s trans %s' % (iname, self.name),
                                     ii*tloc) for iname, ii in ilist]
                # absorbed by this element
                if not np.all(eloc==0):
                    self.iabs_list = [('%s abs %s' % (iname, self.name),
                                       ii*eloc) for iname, ii in ilist]
                # reflected off this element
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
        
    def results(self, filename=None, mode='w', display=True):
        if isarr(self.frequency):
            freq = self.frequency
            if self.single_moded:
                from scipy.constants import c
                conv = np.power(c/freq,2)*1e-9
            else: conv = self.area*1e9
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
        f.write('Element: %s\n' % self.name)
        if self.incident:
            f.write('Source: %s\n' % self.incident.name)
            f.write('Incident power: %s\n' % uprint(self.incident.itrans_int))
            norm = self.incident.itrans_int
        else: norm = self.itrans_int
        f.write('Transmitted power: %s %10.3f%%\n' % 
                (uprint(self.itrans_int), self.itrans_int/norm*100))
        if self.itrans_int:
            for x in self.itrans_list_int:
                f.write('  %-15s %s %10.3f%%\n' % 
                        (x[0].split('emis')[0].strip(), uprint(x[1]),
                         x[1]/self.itrans_int*100))
        f.write('Absorbed power: %s %10.3f%%\n' % 
                (uprint(self.iabs_int), self.iabs_int/norm*100))
        if self.iabs_int:
            for x in self.iabs_list_int:
                tag,rem = x[0].split('emis')
                tag = tag.strip()
                if hasattr(self,'elements'):
                    tag2 = 'to  %-15s' % rem.split('abs')[-1].strip()
                else: tag2 = ''
                f.write('  %-15s%s %s %10.3f%%\n' %
                        (tag, tag2, uprint(x[1]),
                         x[1]/self.iabs_int*100))
        f.write('Reflected power: %s %10.3f%%\n' % 
                (uprint(self.iref_int), self.iref_int/norm*100))
        if self.iref_int:
            for x in self.iref_list_int:
                tag,rem = x[0].split('emis')
                tag = tag.strip()
                if hasattr(self,'elements'):
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
                not isinstance(incident, RadiativeElement):
            raise TypeError,'incident must be an instance of RadiativeElement'
        if isinstance(self.incident,RadiativeElement) and \
                np.all(incident.get_itrans() == self.incident.get_itrans()) \
                and hasattr(self,'itrans'):
            # print 'Stage %s: incident element %s has already be dealt with!' \
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
        print 'Plotting', self.name, suffix.replace('_','')
        
        import pylab
        fig = kwargs.pop('fig', pylab.figure())
        ax = kwargs.pop('ax', pylab.gca())
        
        if x is None: x = self.frequency
        for v,fmt,lab in spectra:
            if isarr(v):
                if ylim is not None:
                    v = parg(v, min(ylim))
                if fmt is not None:
                    ax.plot(x,v,fmt,label=lab)
                else:
                    ax.plot(x,v,label=lab)
        
        if not len(ax.get_lines()):
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
            (trans, '-b', r'$t_{\nu}$'),
            (ref,   '-g', r'$r_{\nu}$'),
            (abs,   '-r', r'$a_{\nu}$')
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
        
        if iitnorm: norm = iitnorm
        elif itnorm: norm = itnorm
        else: norm = np.max([iibnorm, ibnorm, ianorm, irnorm])
        
        spectra = []
        if self.incident:
            iname = self.incident.name.replace(' ',',').upper()
            spectra += [
                (iitrans/norm, '-b', r'$I_{\nu}^{%s}$' % iname),
                ]
            if iibnorm:
                spectra += [
                    (iibb/norm,   '--b', r'$B_{\nu}^{%s}$' % iname),
                    ]
        spectra += [
            (itrans/norm,  '-g', r'$I_{\nu}^{T}$'),
            ]
        if ibb is not None:
            name = self.name.replace(' ',',').upper()
            spectra += [
                (ibb/norm,    '--g', r'$B_{\nu}^{%s}$' % name),
                ]
        if iabs is not None:
            spectra += [
                (iabs/norm,    '-r', r'$I_{\nu}^{A}$'),
                ]
        if iref is not None:
            spectra += [
                (iref/norm,    '-c', r'$I_{\nu}^{R}$')
                ]
        self._plot(spectra, suffix='_spectra', **kwargs)
        
    def plot_trans(self, **kwargs):
        
        if self.incident:
            iitrans, iitnorm = self.incident.get_norm_spec('itrans')
        else: iitnorm = 0.0
        itrans, itnorm = self.get_norm_spec('itrans')
        if iitnorm: norm = iitnorm
        elif itnorm: norm = itnorm
        else: return
        
        spectra = []
        if self.incident:
            spectra += [
                (iitrans/norm, '--k', 'Incident %s' % self.incident.name),
                ]
        spectra += [
            (itrans/norm,  '-k', 'Total Transmitted')
            ]
        for item in self.itrans_list:
            k,v = item
            spectra += [(v/norm, None, k.split('emis')[0].strip())]
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
                (iitrans/norm,  '--k', 'Incident %s' % self.incident.name),
                ]
        
        spectra += [
            (iabs/norm,  '-k', 'Total Absorbed'),
            ]
        for item in self.iabs_list:
            k,v = item
            tag,rem = k.split('emis')
            tag = tag.strip()
            if hasattr(self,'elements'):
                tag += ' to ' + rem.split('abs')[-1].strip()
            spectra += [(v/norm, None, tag)]
        self._plot(spectra, suffix='_abs', ylabel='Absorbed Intensity [a.u.]',
                   **kwargs)
        
    def plot_ref(self, **kwargs):
        
        iref, norm = self.get_norm_spec('iref')
        if not norm: return
        
        spectra = []
        if self.incident:
            iitrans, norm = self.incident.get_norm_spec('itrans')
            spectra += [
                (iitrans/norm,  '--k', 'Incident %s' % self.incident.name),
                ]
        
        spectra += [
            (iref/norm,  '-k', 'Total Reflected'),
            ]
        for item in self.iref_list:
            k,v = item
            tag,rem = k.split('emis')
            tag = tag.strip()
            if hasattr(self,'elements'):
                tag += ' to ' + rem.split('ref')[-1].strip()
            spectra += [(v/norm, None, tag)]
        self._plot(spectra, suffix='_ref', ylabel='Reflected Intensity [a.u.]',
                   **kwargs)
        
class FilterElement(RadiativeElement):
    def __init__(self, name, filt, **kwargs):
        super(FilterElement,self).__init__(name, **kwargs)
        self.filter = filt
        self.trans = filt.get_trans()
        self.abs = filt.get_emis()
        self.ref = filt.get_ref()

class RadiativeStack(RadiativeElement):
    
    def __init__(self, name, elements, incident=None, **kwargs):
        super(RadiativeStack, self).__init__(name, **kwargs)
        self.elements = elements
        self.incident = incident
        if incident is not None: self.propagate(incident)
    
    def propagate(self, incident=None, force=False):
        if self.checkinc(incident, force=force): return
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
        for E in self.elements:
            if E == self.incident:
                curinc = E
                continue
            # propagate incident power to next element
            E.propagate(curinc, force=force)
            # get spectra
            tcur = E.get_trans()
            ecur = E.get_abs()
            rcur = E.get_ref()
            # update absorbed power sourcs
            self.iabs_list.extend(E.get_iabs_list())
            # update reflected power sources. NB: totally reflected power is
            # transmitted twice through previous elements
            rlist = [(rname, rr*self.trans) for rname,rr in E.get_iref_list()]
            self.iref_list.extend(rlist)
            self.ref += rcur * self.trans * self.trans
            # abssivity
            self.abs = self.abs * tcur + ecur
            # transmission
            self.trans *= tcur
            # proceed to next element
            curinc = E
        # update transmitted power sources
        # NB: these have been propagated through all elements
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
                if len(self.elements):
                    self.incident = self.elements[0]
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
        for E in self.elements:
            E.plot_tra(**kwargs)
        return super(RadiativeStack, self).plot_tra(**kwargs)
    
    def plot_spect(self, **kwargs):
        for E in self.elements:
            E.plot_spect(**kwargs)
        return super(RadiativeStack, self).plot_spect(**kwargs)
    
    def plot_trans(self, **kwargs):
        for E in self.elements:
            E.plot_trans(**kwargs)
        return super(RadiativeStack, self).plot_trans(**kwargs)
    
    def plot_abs(self, **kwargs):
        for E in self.elements:
            E.plot_abs(**kwargs)
        return super(RadiativeStack, self).plot_abs(**kwargs)
    
    def plot_ref(self, **kwargs):
        for E in self.elements:
            E.plot_ref(**kwargs)
        return super(RadiativeStack, self).plot_ref(**kwargs)

    def results(self, display_this=True, **kwargs):
        for E in self.elements:
            E.results(**kwargs)
        if display_this:
            return super(RadiativeStack, self).results(**kwargs)
    
###########################################

class SpiderRadiativeModel(object):
    
    # extremal values for Ade filters and nylon
    _T_HP_MIN = 0
    _T_SH_MIN = 0
    _E_HP_MAX = 1
    _E_SH_MAX = 1
    _A_HP_MIN = 0.01
    _A_SH_MIN = 0
    _NY_MIN = 0
    
    # frequency above which the sky is a simple 273K blackbody
    _MAX_FATMOS = 400.0
    # emissivity assumed for 273K atmosphere beyond 1 THz
    _ATMOS_EMIS = 1e-1
    
    def __init__(self,**kwargs):
        self.params = dict()
        self._initialized = False
        profile = kwargs.pop('profile',None)
        self._set_defaults(**kwargs)
        if profile is not None:
            self.load_profile(profile)
        self.set_band(**kwargs)
        self.pretty_print_params()
        self._reload()
        
    def _set_defaults(self,**kwargs):
        # temperatures
        self.params['tsky'] = kwargs.pop('tsky',2.73)
        self.params['esky'] = kwargs.pop('esky',1.0)
        self.params['tvcs1'] = kwargs.pop('tvcs1',35.0)
        self.params['tvcs2'] = kwargs.pop('tvcs2',130.0)
        self.params['t4k'] = kwargs.pop('t4k',5.0)
        self.params['atmos'] = kwargs.pop('atmos',1.0)
        
        # detector quantum efficiency
        self.params['eta'] = kwargs.pop('eta',0.4)
        self.params['esubk'] = kwargs.pop('esubk',0.25) # subk emissivity
        
        self.params['tsubk'] = kwargs.pop('tsubk',0.3)
        
        # window properties
        # emissivity at center freq
        self.params['window_trans'] = kwargs.pop('window_trans',1e-3)
        # temperature
        self.params['Twin'] = kwargs.pop('window_temp',
                                         kwargs.pop('Twin',273.0))
        # emissivity index
        self.params['window_beta'] = kwargs.pop('window_beta',2)
        
        # stop properties
        self.params['t2k'] = kwargs.pop('t2k',kwargs.pop('t_stop',2.0))
        self.params['spill_frac'] = kwargs.pop('spill_frac',0.1)
        
        # nylon conductivity
        self.params['g_nylon'] = 3e-5 # W/K
        
        # aperture diameter
        self.params['aperture'] = 0.3
        
        datdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'rad_data')
        self.params['datdir'] = datdir
        
        figdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'figs')
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        self.params['figdir'] = figdir
        
        self.params['atmfile'] = os.path.join(datdir,'amatm.dat')
        # self.params['spectfile'] = os.path.join(datdir,
        #                                         '145GHzSpectrum.dat')
        self.params['spectfile'] = os.path.join(datdir,
                                                'spectrum_150ghz.dat')
        
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
        print 'Sky temperature: %.3f, emis: %.3f' % (self.params['tsky'],
                                                     self.params['esky'])
        print 'Atmospheric contribution: %.3f' % self.params['atmos']
        print 'VCS2 temperature: %.3f' % self.params['tvcs2']
        print 'VCS1 temperature: %.3f' % self.params['tvcs1']
        print '4K stage temperature: %.3f' % self.params['t4k']
        print 'Stop temperature: %.3f, throughput: %.3f' % \
            (self.params['t2k'],self.params['spill_frac'])
            
    def load_filters(self,norm=True):
        self.filters = {
            'c8-c8': FilterModel('c8-c8','spider_filters_c8-c8.txt',
                                 wavelength=self.wavelength,
                                 t_min=self._T_SH_MIN,
                                 e_max = self._E_SH_MAX,
                                 norm=norm),
            'c12-c16': FilterModel('c12-c16','spider_filters_c12-c16.txt',
                                   wavelength=self.wavelength,
                                   t_min=self._T_SH_MIN,
                                   e_max = self._E_SH_MAX,
                                   norm=norm),
            'c15': FilterModel('c15','spider_filters_c15.txt',
                               wavelength=self.wavelength,
                               t_min=self._T_SH_MIN,
                               e_max=self._E_SH_MAX,
                               norm=norm),
            'c16-c25': FilterModel('c16-c25','spider_filters_c16-c25.txt',
                                   wavelength=self.wavelength,
                                   t_min=self._T_SH_MIN,
                                   e_max = self._E_SH_MAX,
                                   norm=norm),
            'c30': FilterModel('c30','spider_filters_c30.txt',
                               wavelength=self.wavelength,
                               t_min=self._T_SH_MIN,
                               e_max=self._E_SH_MAX,
                               norm=norm),
            '12icm': MetalMeshFilter('12icm',
                                     'spider_filters_w1078_12icm.txt',
                                     wavelength=self.wavelength,
                                     t_min=self._T_HP_MIN,
                                     a_min=self._A_HP_MIN,
                                     e_max = self._E_HP_MAX,
                                     thickness=2.18, norm=norm),
            '7icm': MetalMeshFilter('7icm','spider_filters_w1522_7icm.txt',
                                    wavelength=self.wavelength,
                                    t_min=self._T_HP_MIN,
                                    a_min=self._A_HP_MIN,
                                    e_max = self._E_HP_MAX,
                                    thickness=2.18, norm=norm),
            '4icm': MetalMeshFilter('4icm','spider_filters_4icm.txt',
                                    wavelength=self.wavelength,
                                    t_min=self._T_HP_MIN, a_min=self._A_HP_MIN,
                                    e_max = self._E_HP_MAX,
                                    thickness=2.18, norm=norm),
            '6icm': MetalMeshFilter('6icm','spider_filters_6icm.txt',
                                    wavelength=self.wavelength,
                                    t_min=self._T_HP_MIN, a_min=self._A_HP_MIN,
                                    e_max = self._E_HP_MAX,
                                    thickness=2.18, norm=norm),
            '10icm': MetalMeshFilter('10icm',fcent=8.2,width=1.5,amp=0.93,
                                     wavelength=self.wavelength,
                                     t_min=self._T_HP_MIN,
                                     a_min=self._A_HP_MIN,
                                     thickness=2.18, e_max = self._E_HP_MAX),
            '18icm': MetalMeshFilter('18icm',fcent=17.0,width=2.2,amp=0.93,
                                     wavelength=self.wavelength,
                                     t_min=self._T_HP_MIN,
                                     a_min=self._A_HP_MIN,
                                     thickness=2.18, e_max = self._E_HP_MAX),
            'nylon': NylonFilter(2.38,
                                 wavelength=self.wavelength,
                                 t_min=self._NY_MIN,
                                 norm=norm),
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
        if fcent == 148:
            # self.update_params(
            #     spectfile=os.path.join(self.get_param('datdir'),
            #                            'spectrum_150ghz.dat'))
            self.update_params(
                spectfile=os.path.join(self.get_param('datdir'),
                                       '145GHzSpectrum.dat'))
        elif fcent == 94:
            self.update_params(
                spectfile=os.path.join(self.get_param('datdir'),
                                       'spectrum_90ghz.dat'))
        else:
            raise ValueError,'fcent must be 94 or 148 GHz'
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

    def run_new(self, tag=None, plot=False, interactive=False,
                filter_stack={'vcs2':['c8-c8','c8-c8','c8-c8','c12-c16'],
                              'vcs1':['c12-c16','c16-c25','c16-c25','12icm'],
                              '4k':['10icm','nylon'],
                              '2k':['7icm'],
                              }, **kwargs):
        
        # abscissa and conversion factors
        freq = self.frequency
        freqhz = freq*1.e9
        wlen = self.wavelength
        idx = self.id_band
        from scipy.constants import c
        conv = (c/freq)**2
        npts = len(wlen)
        
        ### TOTAL LOADING
        
        # aperture area
        area = np.pi*(self.params['aperture']/2.)**2
        
        # detector FTS spectrum
        eta = self.params['eta']
        t = self.spectrum
        
        # the sky and window spectrum
        Inu_atmos = self.Inu_atmos
        window_trans = self.params['window_trans'] * \
            (freq/self.params['fcent'])**self.params['window_beta']
        window_trans = threshold(window_trans,high=1.0)
        Inu_win = window_trans*blackbody(freq,self.params['Twin'])
        bb_sky = (blackbody(freq,self.params['tsky'])*self.params['esky']+
                  self.params['atmos']*(Inu_atmos+Inu_win))
        
        # create element
        Rsky = RadiativeElement('Sky', trans=0.0, abs=1.0, bb=bb_sky,
                                wavelength=wlen, frequency=freq,
                                area=area, incident=False)
        
        # assemble the filter stages into RadiativeStack objects
        def make_stack(stage):
            bb = blackbody(freq,self.params['t%s'%stage])
            return RadiativeStack(
                stage.upper(),
                [FilterElement('%s %s %s' % (stage.upper(),chr(ord('A')+i),f),
                               self.filters[f], bb=bb)
                 for i,f in enumerate(filter_stack[stage])])
        
        Rvcs2 = make_stack('vcs2')
        Rvcs1 = make_stack('vcs1')
        R4k = make_stack('4k')
        R2k = make_stack('2k')
        R2k.elements += [
            # hack! spillover from optics sleeve
            RadiativeElement('2K Spillover', bb=R2k.elements[-1].bb,
                             abs=self.params['spill_frac'])
            ]
        
        # sub-K loading (use trans=1 to pass through to detector,
        # but bb=0 to ignore loading onto detector)
        Rsubk = RadiativeElement('sub-K', abs=self.params['esubk'],
                                 ref=1-self.params['esubk'],
                                 )
        
        # detector loading with bandpass
        Rdet = RadiativeElement('Det', trans=t, abs=eta*t,
                                single_moded=True, band=self.id_band2)
        
        # assemble the whole stack and propagate the sky through it
        self.stack = RadiativeStack('TOTAL',
                                    [Rsky, Rvcs2, Rvcs1, R4k, R2k, Rsubk, Rdet],
                                    incident=Rsky)
        
        # second pass to account for loading on nylon
        if any(['nylon' in f for f in filter_stack.values()]):
            g_nylon = self.params['g_nylon']
            if np.isfinite(g_nylon):
                self.stack.results(display=False)
                for E in self.stack.elements:
                    stage = E.name.lower()
                    if stage not in filter_stack: continue
                    if 'nylon' not in filter_stack[stage]: continue
                    for EE in E.elements:
                        if 'nylon' in EE.name:
                            T = self.params['t%s'%stage] + EE.iabs_int / g_nylon
                            print stage,'nylon T=',T,'K'
                            EE.bb = blackbody(freq, T)
                self.stack.propagate(force=True)
        
        # print results
        self.stack.results(display_this=False)
        
        # plot results
        if plot:
            if not interactive:
                import sys
                if 'matplotlib.backends' not in sys.modules:
                    from matplotlib import use
                    use('agg')
            
            figdir = os.path.join(self.params['figdir'],tag)
            if not os.path.exists(figdir): os.mkdir(figdir)
            
            for E in self.stack.elements:
                if E.name == 'Det':
                    pargs = dict(
                        xlim=[0,250],
                        xscale='linear',
                        )
                else:
                    pargs = dict(
                        xlim=None,
                        xscale='log',
                        )
                E.plot_tra(prefix='%s/%s_' % (figdir, tag),
                           ylim=[1e-3,1.1], **pargs)
                E.plot_spect(prefix='%s/%s_' % (figdir, tag),
                             ylim=[1e-8,1.1], **pargs)
                E.plot_trans(prefix='%s/%s_' % (figdir, tag),
                             ylim=[1e-8,1.1], **pargs)
                E.plot_abs(prefix='%s/%s_' % (figdir, tag),
                           ylim=[1e-8,1.1], **pargs)
                E.plot_ref(prefix='%s/%s_' % (figdir, tag),
                           ylim=[1e-8,1.1], **pargs)
        
        return self.stack
        
    def calc_loading(self, stage, i_in, bb_out):
        """
        i_in     incident power onto this stage
        bb_out   stage bb spectrum
        r_in     reflected power off this stage
        i_out    incident power onto next stage
        a_in     power absorbed on this stage
        trans    net transmission
        emis     net emission
        """
        trans = 1.0
        emis = 0.0
        i_out = i_in.copy()
        a_in = 0.0
        r_in = 0.0
        for f in self.filter_stack[stage]:
            tloc = self.filters[f].get_trans()
            eloc = self.filters[f].get_emis()
            rloc = self.filters[f].get_ref()
            a_in += i_out*eloc # absorbed power (assume e=a)
            r_in += i_out*rloc # reflected power
            i_out = i_out*tloc + bb_out*eloc # transmitted/emitted power
            # i_out = i_out*(tloc + eloc)
            trans *= tloc
            emis = emis*tloc + eloc
            # emis += eloc
        # a_in = r_in*emis
        return i_out, r_in, a_in, trans, emis
    
    def calc_components(self, bb_comp, t_comp, e_comp):
        
        i_comp = []
        for b,t,e in zip(bb_comp,t_comp,e_comp):
            i = [ii*t for ii in i_comp[-1]] if len(i_comp) else []
            i.append(b*e)
            i_comp.append(i)
        return i_comp
    
    def run(self,filter_stack={'vcs2':['c8-c8','c8-c8','c8-c8','c12-c16'],
                               'vcs1':['c12-c16','c16-c25','c16-c25','12icm'],
                               '4k':['10icm','nylon'],
                               '2k':['7icm'],
                               },
            tag=None,plot=False,interactive=False):
        
        self.filter_stack = filter_stack
        
        figdir = os.path.join(self.params['figdir'],tag)
        if not os.path.exists(figdir):
            os.mkdir(figdir)
        
        col = {'sky':'b',
               'vcs2':'g',
               'vcs1':'r',
               '4k':'orange',
               '2k':'m',
               'subk':'c'}
        fs = (8,5)
        
        if not plot: interactive=False
        
        freq = self.frequency
        freqhz = freq*1.e9
        wlen = self.wavelength
        idx = self.id_band
        from scipy.constants import c
        conv = (c/freqhz)**2
        npts = len(wlen)
        
        ### TOTAL LOADING
        
        Inu_atmos = self.Inu_atmos
        t = self.spectrum
        
        window_trans = self.params['window_trans'] * \
            (freq/self.params['fcent'])**self.params['window_beta']
        window_trans = threshold(window_trans,high=1.0)
        Inu_win = window_trans*blackbody(freq,self.params['Twin'])
        
        bb_subk = blackbody(freq,self.params['tsubk'])
        bb_2k = blackbody(freq,self.params['t2k'])
        bb_4k = blackbody(freq,self.params['t4k'])
        bb_vcs1 = blackbody(freq,self.params['tvcs1'])
        bb_vcs2 = blackbody(freq,self.params['tvcs2'])
        bb_sky = (blackbody(freq,self.params['tsky'])*self.params['esky']+
                  self.params['atmos']*(Inu_atmos+Inu_win))
        
        i_vcs2 = bb_sky
        i_vcs1, r_vcs2, a_vcs2, t_vcs2, e_vcs2 = \
            self.calc_loading('vcs2', i_vcs2, bb_vcs2)
        i_4k, r_vcs1, a_vcs1, t_vcs1, e_vcs1 = \
            self.calc_loading('vcs1', i_vcs1, bb_vcs1)
        i_2k, r_4k, a_4k, t_4k, e_4k = self.calc_loading('4k', i_4k, bb_4k)
        i_subk, r_2k, a_2k, t_2k, e_2k = self.calc_loading('2k', i_2k, bb_2k)
        e_subk = self.params['esubk']
        r_subk = i_subk*(1-e_subk)
        a_subk = i_subk*e_subk
        
        bb_comp = [bb_sky, bb_vcs2, bb_vcs1, bb_4k, bb_2k]
        t_comp = [np.ones_like(wlen), t_vcs2, t_vcs1, t_4k, t_2k]
        e_comp = [np.ones_like(wlen), e_vcs2, e_vcs1, e_4k, e_2k]
        i_comp = self.calc_components(bb_comp, t_comp, e_comp)
        
        eta = self.params['eta']
        stack = np.prod(t_comp,axis=0)
        stack_max = np.max(stack[idx])
        print ''
        print 'stack transmission: %.3f' % stack_max
        print 'quantum efficiency: %.3f' % eta
        print 'end-to-end transmission: %.3f' % (stack_max*eta)
        
        if plot:
            if not interactive:
                from matplotlib import use
                use('agg')
            import matplotlib.pyplot as plt
            def savefig(figdir,tag1,tag2):
                filename = os.path.join(figdir,'%s%s.png' % 
                                        (tag1,('_%s' % tag2 if tag else '')))
                plt.savefig(filename, bbox_inches='tight')
                if not interactive: plt.close()
                
            def plot_loading(struct,ax=None,xlabel='Wavelength[$\mu$m]',
                             ylabel='Intensity',title=None,
                             xscale='log',yscale='log',
                             xlim=None,ylim=(1e-8,1),legend=True,ncol=1,
                             filetag='intensity',figsize=None):
                if ax is None:
                    if figsize is None: figsize=fs
                    plt.figure(figsize=figsize)
                    ax = plt.gca()
                for data,linestyle,stage,label in struct:
                    color = col[stage] if stage in col else stage
                    ax.plot(wlen,data,linestyle,color=color,label=label)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.set_xscale(xscale)
                    ax.set_yscale(yscale)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    if legend: ax.legend(loc='best',ncol=ncol)
                savefig(figdir,filetag,tag)
            
            plot_loading([(pnorm(bb_sky), '-', 'sky', 'Sky'),
                          (pnorm(bb_vcs2), '-', 'vcs2', 'VCS2'),
                          (pnorm(bb_vcs1), '-', 'vcs1', 'VCS1'),
                          (pnorm(bb_4k), '-', '4k', '4K'),
                          (pnorm(bb_2k), '-', '2k', '2K'),
                          (e_vcs2, '--', 'vcs2', 'VCS2 emis'),
                          (e_vcs1, '--', 'vcs1', 'VCS1 emis'),
                          (e_4k, '--', '4k', '4K emis'),
                          (e_2k, '--', '2k', '2K emis')],
                         title='Emission at each stage',
                         ncol=2, filetag='emission')
            
            plot_loading([(pnorm(i_vcs2), '-', 'vcs2', 'VCS2'),
                          (pnorm(i_vcs1), '-', 'vcs1', 'VCS1'),
                          (pnorm(i_4k), '-', '4k', '4K'),
                          (pnorm(i_2k), '-', '2k', '2K'),
                          (pnorm(i_subk), '-', 'subk', 'sub-K'),
                          (t_vcs2, '--', 'vcs2', 'VCS2 trans'),
                          (t_vcs1, '--', 'vcs1', 'VCS1 trans'),
                          (t_4k, '--', '4k', '4K trans'),
                          (t_2k, '--', '2k', '2K trans')],
                         title='Incident radiation on each stage',
                         ncol=2, filetag='incident_total')
            
            norm = i_vcs2.max()
            plot_loading([(i_vcs2/norm, '-', 'sky', 'Sky'),
                          (t_vcs2, '--', 'k', 'VCS2 trans')],
                         title='Incident radiation on VCS2',
                         filetag='incident_vcs2')
            
            norm = i_vcs1.max()
            plot_loading([(i_vcs1/norm, '-', 'k', 'Total'),
                          (i_comp[1][0]/norm, '-', 'sky', 'Sky'),
                          (i_comp[1][1]/norm, '-', 'vcs2', 'VCS2'),
                          (t_vcs1, '--', 'k', 'VCS1 trans')],
                         title='Incident radiation on VCS1',
                         filetag='incident_vcs1')
            
            norm = i_4k.max()
            plot_loading([(i_4k/norm, '-', 'k', 'Total'),
                          (i_comp[2][0]/norm, '-', 'sky', 'Sky'),
                          (i_comp[2][1]/norm, '-', 'vcs2', 'VCS2'),
                          (i_comp[2][2]/norm, '-', 'vcs1', 'VCS1'),
                          (t_4k, '--', 'k', '4K trans')],
                         title='Incident radiation on 4K',
                         filetag='incident_4k')
            
            norm = i_2k.max()
            plot_loading([(i_2k/norm, '-', 'k', 'Total'),
                          (i_comp[3][0]/norm, '-', 'sky', 'Sky'),
                          (i_comp[3][1]/norm, '-', 'vcs2', 'VCS2'),
                          (i_comp[3][2]/norm, '-', 'vcs1', 'VCS1'),
                          (i_comp[3][3]/norm, '-', '4k', '4K'),
                          (t_2k,'--', 'k', '2K trans')],
                         title='Incident radiation on 2K',
                         filetag='incident_2k')
            
            norm = i_subk.max()
            plot_loading([(i_subk/norm, '-', 'k', 'Total'),
                          (i_comp[4][0]/norm, '-', 'sky', 'Sky'),
                          (i_comp[4][1]/norm, '-', 'vcs2', 'VCS2'),
                          (i_comp[4][2]/norm, '-', 'vcs1', 'VCS1'),
                          (i_comp[4][3]/norm, '-', '4k', '4K'),
                          (i_comp[4][4]/norm, '-', '2k', '2K'),
                          (t,'--', 'k', 'Det')],
                         title='Incident radiation on sub-K',
                         filetag='incident_subk')
        
        ### IN-BAND LOADING
        
        print ''
        print 'Composition of in-band loading:'
        
        idx2 = self.id_band2
        
        spilloverv = t*self.params['spill_frac']*bb_2k*conv
        spillover = int_tabulated(spilloverv[idx2],x=freqhz[idx2],n=npts)
        
        normv = t*i_subk*conv + spilloverv
        norm = int_tabulated(normv[idx2],x=freqhz[idx2],n=npts)
        print 'Total incident [pW]: %.6f' % (norm*1.e12)
        print 'Total absorbed [pW]: %.6f' % (eta*norm*1.e12)
        
        if plot:
            plot_loading([(parg(pnorm(a_vcs2),1e-8), '-', 'vcs2', 'VCS2'),
                          (pnorm(a_vcs1), '-', 'vcs1', 'VCS1'),
                          (pnorm(a_4k), '-', '4k', '4K'),
                          (pnorm(a_2k), '-', '2k', '2K'),
                          (pnorm(a_subk), '-', 'subk', 'sub-K'),
                          (parg(pnorm(normv),1e-8), '-', 'k', 'Det')],
                         # yscale='linear',xscale='linear',
                         title='Loading on each stage',
                         filetag='loading')
        
        xl = 1e-5
        
        if plot:
            plt.figure(figsize=fs)
            ax_ib = plt.gca()
            ax_ib.plot(freq,parg(normv/norm,xl),'k',label='Norm')
        
        arg = i_comp[-1][0]*t*conv
        q_det = int_tabulated(arg[idx2],x=freqhz[idx2],n=npts)
        sky_loading = q_det/norm*100
        print 'Sky: \t\t %9.6f %6.2f%%' % (q_det*1e12, sky_loading)
        if plot:
            ax_ib.plot(freq,parg(arg/norm,xl),color=col['sky'],label='Sky')
        
        arg = i_comp[-1][1]*t*conv
        q_det = int_tabulated(arg[idx2],x=freqhz[idx2],n=npts)
        vcs2_loading = q_det/norm*100
        print 'VCS2: \t\t %9.6f %6.2f%%' % (q_det*1e12, vcs2_loading)
        if plot:
            ax_ib.plot(freq,parg(arg/norm,xl),color=col['vcs2'],label='VCS2')
        
        arg = i_comp[-1][2]*t*conv
        q_det = int_tabulated(arg[idx2],x=freqhz[idx2],n=npts)
        vcs1_loading = q_det/norm*100
        print 'VCS1: \t\t %9.6f %6.2f%%' % (q_det*1e12, vcs1_loading)
        if plot:
            ax_ib.plot(freq,parg(arg/norm,xl),color=col['vcs1'],label='VCS1')
        
        arg = i_comp[-1][3]*t*conv
        q_det = int_tabulated(arg[idx2],x=freqhz[idx2],n=npts)
        lhe_loading = q_det/norm*100
        print '4K stage: \t %9.6f %6.2f%%' % (q_det*1e12, lhe_loading)
        if plot:
            ax_ib.plot(freq,parg(arg/norm,xl),color=col['4k'],label='4K')
        
        arg = i_comp[-1][4]*t*conv
        q_det = int_tabulated(arg[idx2],x=freqhz[idx2],n=npts)
        stop_loading = q_det/norm*100
        print '2K stage: \t %9.6f %6.2f%%' % (q_det*1e12, stop_loading)
        if plot:
            ax_ib.plot(freq,parg(arg/norm,xl),color=col['2k'],label='2K')
        
        stop_loading = spillover/norm*100
        print 'Stop spillover:  %9.6f %6.2f%%' % (spillover*1e12, stop_loading)
        if plot:
            ax_ib.plot(freq,parg(spilloverv/norm,xl),'--',color=col['2k'],
                       label='2K spill')
            
            ax_ib.set_xlabel('Frequency [GHz]')
            ax_ib.set_ylabel('Fraction of Total')
            ax_ib.set_title('In-band Loading')
            # ax_ib.set_xscale('log')
            ax_ib.set_yscale('log')
            ax_ib.set_ylim(xl,5e-2)
            ax_ib.set_xlim(0,250)
            ax_ib.legend(loc='best')
            savefig(figdir,'inband',tag)
            # if not interactive: plt.close()
        
        ### EXCESS LOADING
        
        fudge = 1.0
        area = np.pi*(self.params['aperture']/2.)**2
        
        qa_vcs2 = int_tabulated(a_vcs2*area,x=freqhz,n=npts)
        qa_vcs1 = int_tabulated(a_vcs1*area,x=freqhz,n=npts)
        qa_4k = int_tabulated(a_4k*area,x=freqhz,n=npts)
        qa_2k = int_tabulated(a_2k*area,x=freqhz,n=npts)
        qa_subk = int_tabulated(a_subk*area,x=freqhz,n=npts)
        
        qi_vcs2 = int_tabulated(i_vcs2*area,x=freqhz,n=npts)
        qi_vcs1 = int_tabulated(i_vcs1*area,x=freqhz,n=npts)
        qi_4k = int_tabulated(i_4k*area,x=freqhz,n=npts)
        qi_2k = int_tabulated(i_2k*area,x=freqhz,n=npts)
        qi_subk = int_tabulated(i_subk*area,x=freqhz,n=npts)
        
        q_det = int_tabulated((t*i_subk*conv)[idx2],x=freqhz[idx2],n=npts)
        q_det = eta * (q_det + spillover)
        
        print ''
        print 'Total loading:'
        print 'Power on VCS2 [W]: \t inc %.6e abs %.6e' % (qi_vcs2, qa_vcs2)
        print 'Power on VCS1 [W]: \t inc %.6e abs %.6e' % (qi_vcs1, qa_vcs1)
        print 'Power on 4K [mW]: \t inc %.6e abs %.6e' % (qi_4k*1.e3, qa_4k*1.e3)
        print 'Power on 2K [mW]: \t inc %.6e abs %.6e' % (qi_2k*1.e3, qa_2k*1.e3)
        print 'Power on subK [uW]: \t inc %.6e abs %.6e' % (qi_subk*1.e6, qa_subk*1.e6)
        print 'Power on detector [pW]:  abs %.6e' % (q_det*1.e12)
        
        if interactive:
            plt.show()
        # keyboard()

if __name__ == "__main__":
    
    import argparse as ap
    P = ap.ArgumentParser(add_help=True)
    P.add_argument('model',nargs='?',default=1,type=int)
    P.add_argument('-i','--interactive',default=False,
                   action='store_true',help='show plots')
    P.add_argument('-p','--plot',default=False,
                   action='store_true',help='make plots')
    # P.add_argument('-t','--tag',default='',action='store',
    #                help='plotting tag')
    # P.add_argument('-f','--fcent',default=148,choices=[94,148],
    #                action='store',type=int,
    #                help='detector center frequency')
    args = P.parse_args()
    
    if not args.plot: args.interactive = False
    
    opts = dict()
    fcent = 148
    if args.model == 1:
        filter_stack={'vcs2':['c8-c8','c8-c8','c8-c8','c12-c16'],
                      'vcs1':['c12-c16','c16-c25','c16-c25','12icm'],
                      '4k':['10icm','nylon'],
                      '2k':['7icm'],
                      }
        tag = 'default'
    elif args.model == 2:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm','nylon'],
                      '4k':['10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz'
    elif args.model == 3:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm','nylon'],
                      '4k':['18icm','10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz_18icm'
    elif args.model == 4:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm','nylon'],
                      '4k':['cirlex','10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz_hwp'
    elif args.model == 5:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm','nylon'],
                      '4k':['quartz','10icm','nylon'],
                      '2k':['4icm'],
                      }
        tag = '90ghz_hwp'
        fcent = 94
    elif args.model == 6:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm','nylon'],
                      '4k':['cirlex','10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz_hwp_300k'
        opts = dict(tsky=300,atmos=0)
    elif args.model == 7:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm'],
                      '4k':['10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz_nonylon'
    elif args.model == 8:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm','nylon'],
                      '4k':['10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz_300k'
        opts = dict(tsky=300,atmos=0)
    elif args.model == 9:
        filter_stack={'vcs2':['c15','c15','c30','c30'],
                      'vcs1':['c15','c30','c30','12icm'],
                      '4k':['10icm','nylon'],
                      '2k':['6icm'],
                      }
        tag = '150ghz_300k_nonylon'
        opts = dict(tsky=300,atmos=0)
    else:
        raise ValueError,'unrecognized model number %d' % args.model
    
    # print 'Filter stack:',filter_stack
    print 'Filter stack: %s' % tag
    print 'VCS2:',filter_stack['vcs2']
    print 'VCS1:',filter_stack['vcs1']
    print '4K:',filter_stack['4k']
    print '2K:',filter_stack['2k']
    
    S = SpiderRadiativeModel(fcent=fcent, **opts)
    # S.run(filter_stack=filter_stack, tag=tag,
    #       plot=args.plot, interactive=args.interactive)
    S.run_new(filter_stack=filter_stack, tag=tag,
              plot=args.plot, interactive=args.interactive)
