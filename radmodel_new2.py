from matplotlib import use
use('agg')

import numpy as np
import os
# import reduce since python3 doesn't have reduce buildin
from functools import reduce


from radmodel import *
RadiativeModelOld = RadiativeModel

class TransferMatrix(object):
    """
    Transfer matrix for a single filter.  Dimensions are (2x2xN) for N
    frequency bins.

    Define in terms of t/r/a coefficients (assume reciprocal and symmetric),
    or in terms of an existing transfer matrix.

    If a matrix is supplied, then both the forward and reverse transfer
    coefficients are calculated from it and stored.
    """

    def __init__(self, t=None, r=None, a=None, mat=None, asym=1):
        self.set(t=t, r=r, a=a, mat=mat, inplace=False, asym=asym)

    def set(self, t=None, r=None, a=None, mat=None, inplace=False, asym=1):
        if mat is not None:
            r = mat[:,0,1] / mat[:,1,1]
            t = mat[:,0,0] - mat[:,1,0] * r
            tb = 1. / mat[:,1,1]
            rb = -mat[:,1,0] / mat[:,1,1]
            a = 1. - t - rb
            ab = 1. - tb - r
        else:
            if a is None:
                if r is None:
                    if t is None:
                        t = 1.0
                    r = 1 - t
                a = 1 - r - t
            else:
                if r is None:
                    if t is None:
                        t = 1 - a
                    r = 1 - t - a
                else:
                    if t is None:
                        t = 1 - a - r
            if asym != 1 and np.any([np.size(x) > 1 for x in (t, a)]):
                t, a = np.broadcast_arrays(t, a)
                a = asym * a
                # a += d
                a[a > 1 - t] = 1
                r = 1 - t - a
            tb = t.copy() if not np.isscalar(t) else t
            rb = r.copy() if not np.isscalar(r) else r
            ab = a.copy() if not np.isscalar(a) else a
            # if asym and np.any([np.size(x) > 1 for x in (t, a)]):
            #     t, a = np.broadcast_arrays(t, a)
            #     d = asym * a
            #     a += d
            #     a[a > 1 - t] = 1
            #     rb = 1 - t - a

        eps = 1e-8

        if np.any([np.size(x) > 1 for x in (t,r,a,tb,rb,ab)]):
            t, r, a, tb, rb, ab = np.broadcast_arrays(t, r, a, tb, rb, ab)

        def truncate1(t, r, a, delta=eps):
            b = t > 1
            if np.any(b):
                d = t[b] - 1
                t[b] -= d
                a[b] += d
                r[b] = 1. - a[b] - t[b]
            b = (t < eps)
            if np.any(b) and not np.isscalar(t) and not np.all(t[b] == 0):
                d = t[b]
                t[b] -= d - eps
                a[b] += d - eps
                r[b] = 1. - a[b] - t[b]

        truncate1(a, rb, t)
        truncate1(ab, r, tb)
        truncate1(t, rb, a)
        truncate1(tb, r, ab)

        mat = None

        if mat is None:
            mat = np.asarray([[(t - r*rb/tb), r/tb],[-rb/tb, 1./tb]])
            if mat.ndim > 2:
                mat = mat.transpose(2,0,1)

        if inplace:
            self.t[:] = t
            self.r[:] = r
            self.a[:] = a
            self.tb[:] = tb
            self.rb[:] = rb
            self.ab[:] = ab
            self.mat[:] = mat
        else:
            self.t = t
            self.r = r
            self.a = a
            self.tb = tb
            self.rb = rb
            self.ab = ab
            self.mat = mat

    def __call__(self):
        return self.mat

    @staticmethod
    def _prod(m1, m2):
        return np.matmul(m1, m2)

    def __imul__(self, T):
        if not isinstance(T, self.__class__):
            raise NotImplementedError
        self.set(mat=self._prod(self.mat, T.mat), inplace=True)

    def __mul__(self, T):
        if not isinstance(T, self.__class__):
            raise NotImplementedError
        return self.__class__(mat=self._prod(self.mat, T.mat))

    def get_trans(self, forward=True):
        return self.t if forward else self.tb

    def get_ref(self, forward=True):
        return self.r if forward else self.rb

    def get_abs(self, forward=True):
        return self.a if forward else self.ab

    def get_tra(self, forward=True):
        if forward:
            return self.t, self.rb, self.a
        return self.tb, self.r, self.ab

    def copy(self):
        return self.__class__(mat=self.mat.copy())

class RadiativeSurface(object):

    def __init__(self, name, temperature=None, frequency=None,
                 bb=0, trans=1.0, abs=0.0, ref=0.0, asym=1,
                 wavelength=None, aperture=None, area=None,
                 antenna=False, band=None, verbose=False,
                 spill_frac=0, **kwargs):
        self.verbose = verbose
        if self.verbose:
            print('Initializing surface',name)
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
        if aperture:
            self.area = np.pi*(aperture/2.)**2
        else:
            self.area = area
        self.spill_frac = spill_frac

        self.set_xfer(trans=trans, ref=ref, abs=abs, asym=asym)
        self.done = False
        self.init()

    def __repr__(self):
        return "%s('%s')" % (self.__class__.__name__, self.name)

    def set_xfer(self, trans=1.0, ref=0.0, abs=0.0, asym=1):
        self.xfer = TransferMatrix(t=trans, r=ref, a=abs, asym=asym)

    def get_xfer(self):
        return self.xfer

    def get_subxfer(self, idx1, idx2):
        return self.xfer

    def get_trans(self, forward=True):
        """Transmission spectrum"""
        return self.xfer.get_trans(forward=forward)

    def get_ref(self, forward=False):
        """Reflection spectrum"""
        return self.xfer.get_ref(forward=forward)

    def get_abs(self, forward=True):
        """Emission spectrum"""
        return self.xfer.get_abs(forward=forward)

    def set_emis(self):
        for forward in [True, False]:
            attr = 'fwd_emis' if forward else 'rev_emis'
            if self.check_attr(attr):
                continue

            a = self.get_abs(forward=not forward)
            if not ( np.all(self.bb == 0) or np.all(a == 0) ):
                emis = a * self.bb
                self.add_to_attr(attr, emis)
            else:
                self.done_attr(attr)

    def get_spect(self, stype, forward=True, source=None, integrate=False,
                  aslist=False):
        attr = stype if stype in ['delta', 'extra'] else \
               ('fwd_{}' if forward else 'rev_{}').format(stype)
        attr = ('int_{}' if integrate else 'spect_{}').format(attr)
        attrint = attr
        attrlist = '{}_list'.format(attr)
        if aslist:
            attr = attrlist
        if source is None:
            return getattr(self, attr)
        attr = attrint

        slist = getattr(self, attrlist)
        if len(slist):
            stot = 0
            for n, s in slist:
                if source.lower() in n.split(stype)[0].strip().lower():
                    stot = stot + s
            if not np.all(stot == 0):
                return stot
        print(source.lower())
        print([x[0].split(stype)[0].strip().lower() for x in slist])
        raise KeyError( 'source {} not found in {}'.format(source, attr))

    def get_ispect(self, stype, **kwargs):
        return self.get_spect(stype, integrate=True, **kwargs)

    # TODO option to normalize
    # TODO option to return integrals
    # TODO use OrderedDict everywhere instead of lists

    def get_emis(self, **kwargs):
        return self.get_spect('emis', **kwargs)

    def get_inc(self, **kwargs):
        return self.get_spect('inc', **kwargs)

    def get_itrans(self, **kwargs):
        return self.get_spect('trans', **kwargs)

    def get_iref(self, **kwargs):
        return self.get_spect('ref', **kwargs)

    def get_iabs(self, total=False, forward=True, **kwargs):
        if total:
            return self.get_spect('abs', forward=True, **kwargs) + \
                self.get_spect('abs', forward=False, **kwargs)
        return self.get_spect('abs', forward=forward, **kwargs)

    def get_delta(self, **kwargs):
        return self.get_spect('delta', **kwargs)

    def get_extra(self, **kwargs):
        return self.get_spect('extra', **kwargs)

    def propagate(self, spect, source=None, forward=True):
        pdir = 'fwd' if forward else 'rev'
        ndir = 'rev' if forward else 'fwd'
        st = spect * self.get_trans(forward=forward)
        sr = spect * self.get_ref(forward=not forward)
        sa = spect * self.get_abs(forward=forward)
        prop = [('{}_inc'.format(pdir), spect),
                ('{}_trans'.format(pdir), st),
                ('{}_ref'.format(ndir), sr),
                ('{}_abs'.format(pdir), sa)]
        for attr, spec in prop:
            self.add_to_attr(attr, spec, source=source)

    def init_attr(self, attr):
        setattr(self, 'spect_%s_list' % attr, [])
        setattr(self, 'spect_%s' % attr, 0.0)
        setattr(self, 'int_%s_list' % attr, [])
        setattr(self, 'int_%s' % attr, 0.0)
        setattr(self, 'check_%s' % attr, False)

    def init(self):
        if self.done:
            return
        self.init_attr('fwd_inc')
        self.init_attr('fwd_trans')
        self.init_attr('fwd_ref')
        self.init_attr('fwd_abs')
        self.init_attr('fwd_emis')
        self.init_attr('rev_inc')
        self.init_attr('rev_trans')
        self.init_attr('rev_ref')
        self.init_attr('rev_abs')
        self.init_attr('rev_emis')
        self.init_attr('delta')
        self.init_attr('extra')
        self.set_emis()
        self.done = True

    def check_attr(self, attr):
        return getattr(self, 'check_%s' % attr)

    def done_attr(self, attr):
        setattr(self, 'check_%s' % attr, True)

    def integrate(self, spect):
        freq = self.frequency * 1e9 # hz
        if self.antenna:
            from scipy.constants import c
            conv = np.power(c/freq, 2)
        else:
            conv = self.area*np.pi # Lambertian scattering!
        sint = integrate(conv * spect, freq, idx=self.band)
        return sint

    def add_to_attr(self, attr, spect, source=None):
        if np.all(spect==0):
            return

        stype = None
        for t in ['inc', 'emis', 'abs', 'trans', 'ref', 'delta', 'extra']:
            if t in attr:
                stype = t
                break
        if not stype:
            raise KeyError('unrecognized attr %s' % attr)

        if stype in ['delta', 'extra']:
            if source is None:
                raise ValueError('source required for delta or extra')
            tag = source
        else:
            if source is None:
                tag = '%s %s' % (self.name, stype)
            else:
                tag = '%s %s %s' % (source, stype, self.name)
            if stype in ['emis', 'abs', 'ref'] and attr != 'fwd_ref':
                # add to heat transfer
                f = -1 if stype == 'emis' else 1
                if stype != 'ref':
                    self.add_to_attr('delta', spect * f, source=tag)
                else:
                    f *= self.spill_frac
                self.add_to_attr('extra', spect * f, source=tag)

        if self.verbose:
            print('Surface {}: Adding `{}` to `{}`, {}'.format(
                self.name, tag, attr, uprint(self.integrate(spect))))

        spect_list = getattr(self, 'spect_%s_list' % attr)
        tag_list = [x[0] for x in spect_list]
        if tag in tag_list:
            idx = tag_list.index(tag)
            t, s = spect_list[idx]
            s += spect
            spect_list[idx] = (t, s)
        else:
            spect_list += [(tag, spect.copy())]
        setattr(self, 'spect_%s_list' % attr, spect_list)

        spect_tot = getattr(self, 'spect_%s' % attr)
        spect_tot += spect
        setattr(self, 'spect_%s' % attr, spect_tot)

        if isarr(self.frequency):
            sint = self.integrate(spect)

            int_list = getattr(self, 'int_%s_list' % attr)
            int_tot = getattr(self, 'int_%s' % attr)
            if tag in tag_list:
                t, s = int_list[idx]
                s += sint
                int_list[idx] = (t, s)
            else:
                int_list += [(tag, sint)]
            int_tot += sint
            setattr(self, 'int_%s_list' % attr, int_list)
            setattr(self, 'int_%s' % attr, int_tot)

        self.done_attr(attr)

    def get_attr_dict(self, attr):
        keys = ['spect_{}_list', 'spect_{}', 'int_{}_list', 'int_{}', 'check_{}']
        keys = [x.format(attr) for x in keys]
        adict = {}
        for k in keys:
            adict[k] = getattr(self, k)
        return adict

    def set_attr_dict(self, attr, adict):
        for k, v in adict.iteritems():
            setattr(self, k, v)

    def copy_attr(self, attr, source, append=False):

        S = self if append else source
        adict = S.get_attr_dict(attr)

        if append:
            # not in-place!
            sdict = source.get_attr_dict(attr)
            for k in adict.keys():
                if k.startswith('check'):
                    adict[k] = adict[k] | sdict[k]
                else:
                    adict[k] = adict[k] + sdict[k]

        self.set_attr_dict(attr, adict)

    def append_attr(self, attr, source):
        self.copy_attr(self, attr, source, append=True)

    def results(self, filename=None, mode='w', display=True, summary=False):

        if not display:
            return

        if filename is None:
            import sys
            f = sys.stdout
        else:
            if isinstance(filename, file):
                f = filename
            else:
                f = open(filename, mode)

        def write_group(name, attr, norm=None, split=None):
            if norm == 0:
                norm = None
            intattr = getattr(self, 'int_{}'.format(attr))
            intlist = getattr(self, 'int_{}_list'.format(attr))
            if norm is None:
                f.write('%-12s: %s\n' % (name, uprint(intattr)))
            else:
                f.write('%-12s: %s %10.3f%%\n' %
                        (name, uprint(intattr), intattr / norm * 100))
            if not summary and intattr != 0:
                for k,v in intlist:
                    if v == 0:
                        continue
                    if split is not None:
                        if not isinstance(split, list):
                            split = [split]
                        for ss in split:
                            k = k.split(' {}'.format(ss.strip()))[0].strip()
                    f.write('  %-15s %s %10.3f%%\n' %
                            (k, uprint(v), v / intattr * 100))

        f.write('\n'+'*'*80+'\n')
        f.write('%-8s: %s (%s)\n' % ('Surface', self.name, tprint(self.temperature)))
        tf = self.get_trans(forward=True)
        tr = self.get_trans(forward=False)
        if np.size(tf) > 1:
            cf = self.frequency[::-1][np.argmin(np.abs(tf[::-1] - 0.5))]
            cr = self.frequency[::-1][np.argmin(np.abs(tr[::-1] - 0.5))]
            ucf = uprint(cf * 1e9, unit='Hz', fmt='%8.3f')
            ucr = uprint(cf * 1e9, unit='Hz', fmt='%8.3f')
            f.write('%-12s: %-15s: %s\n' % (self.name, 'Forward cutoff', ucf))
            f.write('%-12s: %-15s: %s\n' % (self.name, 'Reverse cutoff', ucr))

        write_group('FWD EMISSION', 'fwd_emis', split='emis')
        write_group('FWD INCIDENT', 'fwd_inc', split=['inc','emis'])
        if self.int_fwd_inc > 0:
            norm = self.int_fwd_inc
        else:
            norm = self.int_fwd_emis

        write_group('FWD ABSORB', 'fwd_abs', norm=norm, split=['abs','emis'])
        write_group('FWD TRANSMIT', 'fwd_trans', norm=norm, split=['trans','emis'])
        write_group('REV REFLECT', 'rev_ref', norm=norm, split=['ref','emis'])

        f.write('-'*50+'\n')
        write_group('REV EMISSION', 'rev_emis', split='emis')
        write_group('REV INCIDENT', 'rev_inc', split=['inc','emis'])
        if self.int_rev_inc > 0:
            norm = self.int_rev_inc
        else:
            norm = self.int_rev_emis

        write_group('REV ABSORB', 'rev_abs', norm=norm, split=['abs','emis'])
        write_group('FWD REFLECT', 'fwd_ref', norm=norm, split=['ref','emis'])
        write_group('REV TRANSMIT', 'rev_trans', norm=norm, split=['trans','emis'])

        f.write('-'*50+'\n')
        write_group('NET TRANSFER', 'delta', split=['emis', 'abs'])
        write_group('EXT TRANSFER', 'extra', split=['emis', 'abs', 'ref'])

        if filename is not None and not isinstance(filename, file):
            f.close()

    def get_norm_spec(self, attr):
        if not hasattr(self, attr):
            return None, 0
        spec = getattr(self, attr)
        if np.isscalar(spec):
            return None, 0
        if not isinstance(spec, np.ndarray):
            return None, 0
        return spec, spec.max()

    def _plot(self, spectra, x=None, prefix='', suffix='',
              xlim=None, ylim=None, xscale='log', yscale='log',
              **kwargs):

        if self.verbose:
            print('Plotting', self.name, suffix.replace('_',''))

        import pylab
        fig = kwargs.pop('fig', pylab.figure())
        ax = kwargs.pop('ax', pylab.gca())

        line_cycle = ['-','--','-.',':']
        from matplotlib import rcParams
        nc = len(rcParams['axes.color_cycle'])

        line_count = 0

        if x is None:
            x = self.frequency
        for v,fmt,lab,kw in spectra:
            if isarr(v):
                if ylim is not None:
                    v = parg(v, min(ylim))
                _, v = np.broadcast_arrays(x, v)
                if fmt is not None:
                    ax.plot(x,v,fmt,label=lab,**kw)
                else:
                    fmt = line_cycle[int(np.floor(line_count/nc))]
                    ax.plot(x,v,fmt,label=lab,**kw)
                    line_count += 1

        if not len(ax.get_lines()):
            if self.verbose:
                print('No data!')
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

    def plot_tra(self, forward=True, **kwargs):

        trans = self.get_trans(forward=forward)
        abs = self.get_abs(forward=forward)
        ref = self.get_ref(forward=not forward)

        spectra = [
            (trans, '-b', r'$t_{\nu}$', {}),
            (ref,   '-g', r'$r_{\nu}$', {}),
            (abs,   '-r', r'$a_{\nu}$', {})
            ]
        suffix = '_coeff_{}'.format('fwd' if forward else 'rev')
        self._plot(spectra, ylabel='Coefficient', suffix=suffix, **kwargs)

class FilterSurface(RadiativeSurface):
    def __init__(self, name, filt, asym=1, **kwargs):
        self.filter = filt
        t = filt.get_trans()
        r = filt.get_ref()
        a = filt.get_emis()
        asym = asym if (filt.type == 'partial') and ('icm' not in name) else 1
        super(FilterSurface, self).__init__(
            name, trans=t, ref=r, abs=a, asym=asym, **kwargs)

class RadiativeStack(RadiativeSurface):

    def __init__(self, name, surfaces, **kwargs):
        self.surfaces = surfaces
        from collections import OrderedDict
        self.surfdict = OrderedDict((x.name.lower(),x) for x in self.surfaces)
        super(RadiativeStack, self).__init__(name, **kwargs)

    def init(self):
        for S in self.surfaces:
            S.init()
        super(RadiativeStack, self).init()

    def set_xfer(self, **kwargs):
        tmats = [S.xfer for S in self.surfaces]
        # right-multiply
        self.xfer = reduce(lambda x,y: y * x, tmats, TransferMatrix())
        self.set_qmat()
        self.set_gmat()

    def set_qmat(self):
        n = len(self.surfaces) + 1
        T = np.zeros((len(self.frequency), n, n))
        R = np.zeros_like(T)
        TB = np.zeros_like(T)
        RB = np.zeros_like(T)
        for k, S in enumerate(self.surfaces):
            t, rb, a = S.xfer.get_tra(forward=True)
            tb, r, ab = S.xfer.get_tra(forward=False)
            T[:, k+1, k] = t
            TB[:, k, k+1] = tb
            R[:, k+1, k+1] = r
            RB[:, k, k] = rb
        self.qmat = np.asarray([[T, R], [RB, TB]])

    def set_gmat(self):
        # quadrants of transport matrix
        ((T, R), (RB, TB)) = self.qmat
        n = len(self.surfaces) + 1
        nf = len(self.frequency)

        def mmul(x, y):
            return np.matmul(x, y)

        # inv(1 - TB)
        d = np.tile(np.eye(n), (nf, 1, 1)) - TB
        dinv = np.linalg.inv(d)

        # intermediate steps
        Rdinv = mmul(R, dinv)
        dinvRB = mmul(dinv, RB)

        # quadrants of Green's matrix
        Ainv = np.tile(np.eye(n), (nf, 1, 1)) - T - mmul(Rdinv, RB)
        A = np.linalg.inv(Ainv)
        B = mmul(A, Rdinv)
        C = mmul(dinvRB, A)
        D = dinv + mmul(C, Rdinv)

        # store
        self.gmat = np.asarray([[A, B],[C, D]])
        self.delta_gmat = np.asarray([A - C, B - D])

    def propagate(self, spect, source=None, forward=True, index=None):

        if np.all(spect==0):
            return

        # solve for all input / output terms
        n = len(self.surfaces)
        names = [S.name for S in self.surfaces]
        if index is None:
            if source not in names:
                # incident radiation
                index = 0 if forward else n
            else:
                # internal emission
                index = names.index(source)
                if forward:
                    index += 1

        if self.verbose:
            print('Stack {}: found {} source {} at index {}, {}'.format(
                self.name, 'fwd' if forward else 'rev', source, index,
                uprint(self.integrate(spect))))

        # extract appropriate column of G for the given input
        ((A, B), (C, D)) = self.gmat
        if forward:
            solf = A[:, :, index] * spect[:, None]
            solr = C[:, :, index] * spect[:, None]
        else:
            solf = B[:, :, index] * spect[:, None]
            solr = D[:, :, index] * spect[:, None]

        # propagate internally
        for k, S in enumerate(self.surfaces):
            S.propagate(solf[:, k], source=source, forward=True)
            S.propagate(solr[:, k+1], source=source, forward=False)

        # propagate incident terms
        if index in [0, n] and source not in names:
            super(RadiativeStack, self).propagate(
                spect, source=source, forward=forward)

    def set_emis(self):
        if self.check_attr('fwd_emis') and self.check_attr('rev_emis'):
            return

        # propagate internal terms
        for k, S in enumerate(self.surfaces):
            if k < len(self.surfaces) - 1:
                for src, spect in S.get_emis(forward=True, aslist=True):
                    self.propagate(spect, source=src, forward=True, index=k+1)
            if k > 0:
                for src, spect in S.get_emis(forward=False, aslist=True):
                    self.propagate(spect, source=src, forward=False, index=k)

        # total emission
        for forward in [True, False]:
            if forward:
                attr = 'fwd_emis'
                S = self.surfaces[-1]
            else:
                attr = 'rev_emis'
                S = self.surfaces[0]

            for src, spect in S.get_itrans(forward=forward, aslist=True):
                src = src.split(' trans')[0].split(' ref')[0]
                src = src.split(' emis')[0].strip()
                self.add_to_attr(attr, spect, source=src)
            for src, spect in S.get_emis(forward=forward, aslist=True):
                src = src.split(' trans')[0].split(' ref')[0]
                src = src.split(' emis')[0].strip()
                self.add_to_attr(attr, spect, source=src)

            # emission from outmost layer
            # E = S.get_emis(forward=forward) + \
            #     S.get_itrans(forward=forward)
            # self.add_to_attr(attr, E)

            # TODO handle separate emission
            # for (n, E) in S.get_emis(forward=forward, aslist=True):
            #     self.add_to_attr(attr, E, source=S.name)
            # for (n, E) in S.get_itrans(forward=forward, aslist=True):
            #     self.add_to_attr(attr, E, source=S.name)

            self.done_attr(attr)

    # TODO
    # label each term
    # nested stuff should work... inner surfaces will be fractions of stack?
    # replace trans --> in, ref --> out for plotting/etc
    # check that A + T + R = 1 everywhere...
    def plot_tra(self, summary=False, **kwargs):
        if not summary:
            for S in self.surfaces:
                S.plot_tra(**kwargs)
        return super(RadiativeStack, self).plot_tra(**kwargs)

    def plot_spect(self, summary=False, **kwargs):
        if not summary:
            for S in self.surfaces:
                S.plot_spect(**kwargs)
        return super(RadiativeStack, self).plot_spect(**kwargs)

    def plot_trans(self, summary=False, **kwargs):
        if not summary:
            for S in self.surfaces:
                S.plot_trans(**kwargs)
        return super(RadiativeStack, self).plot_trans(**kwargs)

    def plot_abs(self, summary=False, **kwargs):
        if not summary:
            for S in self.surfaces:
                S.plot_abs(**kwargs)
        return super(RadiativeStack, self).plot_abs(**kwargs)

    def plot_ref(self, summary=False, **kwargs):
        if not summary:
            for S in self.surfaces:
                S.plot_ref(**kwargs)
        return super(RadiativeStack, self).plot_ref(**kwargs)

    def results(self, display_this=True, summary=False, **kwargs):
        if display_this:
            super(RadiativeStack, self).results(summary=summary, **kwargs)
        if not summary or (summary and not display_this):
            for S in self.surfaces:
                S.results(summary=summary, **kwargs)

###########################################

class RadiativeModel(RadiativeModelOld):

    def set_defaults(self, **kwargs):
        # kwargs.setdefault('det_floor', 0)

        super(RadiativeModel, self).set_defaults(**kwargs)

        # extra spillover
        self.params['asym_win'] = kwargs.pop('asym_win', 1)
        self.params['asym_vcs1'] = kwargs.pop('asym_vcs1', 1)
        self.params['asym_vcs2'] = kwargs.pop('asym_vcs2', 1)
        self.params['spill_frac_vcs1'] = kwargs.pop('spill_frac_vcs1', 0.0)
        self.params['spill_frac_vcs2'] = kwargs.pop('spill_frac_vcs2', 0.0)
        self.params['spill_frac_win'] = kwargs.pop('spill_frac_win', 0.0)

    def run(self, tag=None, plot=False, interactive=False, summary=False,
            display=False,
            filter_stack={'window': ['poly_window'],
                          'vcs2':['c8-c8','c8-c8','c8-c8','c12-c16'],
                          'vcs1':['c12-c16','c16-c25','c16-c25','12icm'],
                          '4k':['10icm','nylon'],
                          '2k':['7icm'],
                          },
            filter_offsets={}, **kwargs):

        # self.set_defaults(**kwargs)
        self.params.update(kwargs)

        # abscissa and conversion factors
        freq = self.frequency
        wlen = self.wavelength

        # aperture area
        area = np.pi*(self.params['aperture']/2.)**2

        opts = dict(wavelength=wlen, frequency=freq,
                    area=area, verbose=self.verbose)

        surfaces = []
        if self.params['atmos']:
            Ratmos = RadiativeSurface('Atmosphere', trans=self.t_atmos,
                                      abs=self.e_atmos, bb=self.bb_atmos,
                                      **opts)
            surfaces.append(Ratmos)

        # assemble the filter stages into RadiativeStack objects
        def make_stack(stage):
            T = self.params['t%s'%stage]
            tdict = filter_offsets.get(stage, {})
            flist = filter_stack[stage]
            stack = []
            asym = self.params.get('asym_%s'%stage, 1)
            sfrac = self.params.get('spill_frac_%s'%stage, 0)
            for i,f in enumerate(flist):
                dt = tdict.get(f,0)
                bb = blackbody(freq, T+dt)
                tag = ''
                idx = np.where(np.array(flist)==f)[0]
                if len(idx)>1:
                    tag = ' ' + chr(ord('A')+list(idx).index(i))
                if i == 0 or i == len(flist) - 1:
                    asymf = 1
                else:
                    asymf = asym
                S = FilterSurface('%s %s%s' % (stage.upper(), f, tag),
                                  self.filters[f], bb=bb, asym=asymf,
                                  temperature=T+dt, **opts)
                stack.append(S)
            return RadiativeStack(stage.upper(), stack, temperature=T,
                                  spill_frac=sfrac, **opts)

        # window
        if self.params['window']:
            Rwin = make_stack('window')
            surfaces.append(Rwin)

        Rvcs2 = make_stack('vcs2')
        Rvcs1 = make_stack('vcs1')
        R4k = make_stack('4k')
        R2k = make_stack('2k')
        surfaces += [Rvcs2, Rvcs1, R4k, R2k]

        # sub-K loading (use trans=1 to pass through to detector,
        # but bb=0 to ignore loading onto detector)
        Rsubk = RadiativeSurface('Sub-K', **opts)
        surfaces.append(Rsubk)

        # detector loading with bandpass
        eta = self.params['eta']
        t = self.spectrum # detector FTS spectrum
        # Rband = RadiativeSurface('Bandpass', trans=t, abs=1-t,
        #                          antenna=True, band=self.id_band,
        #                          **opts)
        # surfaces.append(Rband)
        Rdet = RadiativeSurface('Det', trans=1-eta*t, abs=eta*t,
                                antenna=True, **opts)
        surfaces.append(Rdet)

        # TODO: parallel surfaces
        # stack: [1, 2, [3, 4]] -- 3 and 4 are parallel!
        # how to construct the matrix?

        # sub-K loading
        # Rsubk = RadiativeSurface('Sub-K', abs=self.params['esubk'],
        #                          ref=1 - self.params['esubk'] - 1e-15, trans=1e-15,
        #                          bb=blackbody(freq, self.params['tsubk']),
        #                          **opts)
        # surfaces.append(Rsubk)

        # assemble the whole stack
        self.tag = tag
        self.stack = RadiativeStack('TOTAL', surfaces, **opts)
        names = [x.name for x in surfaces]

        # sky is a source! add to initial conditions
        spect = self.params['esky'] * blackbody(freq, self.params['tsky'])
        self.stack.propagate(spect, source='Sky', forward=True)

        # spillover is an extra source from 2K! add to initial conditions
        spill = self.params['spill_frac']
        if spill:
            spect = spill * R2k.surfaces[-1].bb
            index = names.index('2K')+1
            self.stack.propagate(spect, source='2K Stop',
                                 index=index, forward=True)

        # extra spillover terms
        # spill_vcs2 = self.params['spill_frac_vcs2']
        # if spill_vcs2:
        #     spect = spill * Rvcs2.surfaces[-1].bb
        #     index = names.index('VCS2')+1
        #     self.stack.propagate(spect, source='VCS2 Spillover',
        #                          index=index, forward=True)

        # spill_vcs1 = self.params['spill_frac_vcs1']
        # if spill_vcs1:
        #     spect = spill * Rvcs1.surfaces[-1].bb
        #     index = names.index('VCS1')+1
        #     self.stack.propagate(spect, source='VCS1 Spillover',
        #                          index=index, forward=True)

        # spill_win = self.params['spill_frac_win']
        # if spill_win:
        #     spect = spill * Rwin.surfaces[-1].bb
        #     index = names.index('WINDOW')+1
        #     self.stack.propagate(spect, source='Window Spillover',
        #                          index=index, forward=True)

        # sub-K loading is an extra source from sub-K?
        # spect = self.params['esubk'] * blackbody(freq, self.params['tsubk'])
        # index = names.index('2K')+1
        # self.stack.propagate(spect, source='Focal Plane',
        #                      index=index, forward=True)

        # print results
        self.results(tag=tag if not display else None,
                     summary=summary, display=display)

        # plot results
        if plot:
            self.plot(tag=tag, interactive=interactive)
        return self.stack

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
        if not os.path.exists(figdir):
            os.mkdir(figdir)

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

            S.plot_tra(ylim=[1e-8,1.1], forward=True, **pargs)
            S.plot_tra(ylim=[1e-8,1.1], forward=False, **pargs)
            # S.plot_spect(ylim=[1e-8,1.1], **pargs)
            # S.plot_trans(ylim=[1e-8,1.1], **pargs)
            # S.plot_abs(ylim=[1e-8,1.1], **pargs)
            # S.plot_ref(ylim=[1e-8,1.1], **pargs)

    def results(self, tag=None, summary=False, display=True):

        if tag is not None:
            figdir = self.params.get('figdir', '.')
            if not os.path.exists(figdir):
                os.mkdir(figdir)
            filename = '{}/{}.txt'.format(figdir, tag)
            f = open(filename, 'w')
            display = True
        else:
            import sys
            f = sys.stdout

        if display:
            f.write('*'*80+'\n')
            if hasattr(self,'tag') and self.tag:
                f.write('MODEL: {}\n'.format(self.tag.replace('_',' ')))
            f.write('*'*80+'\n')
            self.pretty_print_params(filename=f)
        self.stack.results(display_this=False, display=display,
                           summary=summary, filename=f)

        if tag is not None:
            f.close()

def filter_load(obj, t2k=2, t4k=5, tvcs1=35, tvcs2=130, twin=300,
                n_inserts=6, verbose=False, tag=None, filename=None,
                extra=False, mode='w', **params):
				
    if isinstance(obj, RadiativeStack):
        stack = obj
    else:
        stack = obj.run(display=verbose, plot=False, twin=twin,
                        tvcs2=tvcs2, tvcs1=tvcs1, t4k=t4k, t2k=t2k,
                        **params)

    def get_delta(surf):
        if extra:
            return getattr(surf, 'int_extra') * n_inserts
        return getattr(surf, 'int_delta') * n_inserts

    surfs = stack.surfdict
    window_VCS2 = get_delta(surfs['vcs2'])
    if np.isnan(window_VCS2):
        raise ValueError('NaN!')
    window_VCS1 = get_delta(surfs['vcs1'])
    if np.isnan(window_VCS1):
        raise ValueError('NaN!')
    window_MT = get_delta(surfs['4k'])
    if np.isnan(window_MT):
        raise ValueError('NaN!')
    inband = surfs['det'].int_fwd_abs

    if verbose:
        if filename is None:
            import sys
            f = sys.stdout
        else:
            if not isinstance(filename, file):
                f = open(filename, mode)
            else:
                f = filename

        if tag:
            if extra:
                tag = '%s_extra' % tag
            f.write(tag+'\n')
        f.write(' %-10s: %s' % ('In-band', uprint(inband))+'\n')
        f.write(' %-10s: %s' % ('4K', uprint(window_MT))+'\n')
        f.write(' %-10s: %s' % ('VCS1', uprint(window_VCS1))+'\n')
        f.write(' %-10s: %s' % ('VCS2', uprint(window_VCS2))+'\n')
        if filename is not None and not isinstance(filename, file):
            f.close()

    return inband, window_MT, window_VCS1, window_VCS2

if __name__ == "__main__":

    main(model_class=RadiativeModel)
