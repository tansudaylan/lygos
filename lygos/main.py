import os, datetime, fnmatch, time as timemodu

import numpy as np

import scipy.interpolate
from scipy import ndimage

#from numba import jit, prange

import h5py

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from astroquery.mast import Catalogs
import astroquery.mast
import astroquery

import astropy
import astropy.wcs

import miletos

import tdpy
from tdpy.util import summgene

import nicomedia
import miletos


def down_tcut(gdat):
            
    #strgsrch = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    #print('Calling TESSCut at %s with size %d to get the data...' % (strgsrch, gdat.numbside[p]))
    #listhdundatatemp = astroquery.mast.Tesscut.get_cutouts(coordinates=strgsrch, size=gdat.numbside[p])

    print('')
    print('Will download the data via wget, unzip, and read the files...')
    
    strgfile = 'astrocut?ra=%.6f&dec=%.6f&y=%d&x=%d' % (gdat.rasctarg, gdat.decltarg, gdat.numbsidedefa, gdat.numbsidedefa)
    strgfileastrzipp = '%sastrocut_%s.zip' % (gdat.pathdatatargtcut, strgfile)
    
    cmnd = 'wget "https://mast.stsci.edu/tesscut/api/v0.1/%s" -O "%s"' % (strgfile, strgfileastrzipp)
    print(cmnd)
    os.system(cmnd)
    
    cmnd = 'tar -zxvf "%s" -C %s' % (strgfileastrzipp, gdat.pathdatatargtcut)
    print(cmnd)
    os.system(cmnd)


def retr_fluxfromtmag(tmag):
    '''
    Calculate the flux per second from the TESS magnitude
    '''
    
    flux = 10**(0.4 * (20.4 - tmag)) # [e-/s]
    
    return flux


def retr_tmagfromflux(flux):
    '''
    Calculate the TESS magnitude from flux per second
    '''
    
    tmag = 20.4 - 2.5 * np.log10(flux)
    
    return tmag


def read_psfntess(a, b, k, l):
    
    indxcams = range(1, 5)
    indxccds = range(1, 5)
    indxrows = np.array([1, 513, 1025, 1536, 2048])
    indxcols = np.array([45, 557, 1069, 1580, 2092])
    
    indxrowsthis = np.argmin(abs(k - indxrows))
    indxcolsthis = np.argmin(abs(l - indxcols))
    
    pathdata = os.environ['LYGOS_DATA_PATH'] + '/tesspsfn/'
    pathsubs = pathdata + 'tess_prf-master/cam%d_ccd%d/' % (a, b)
    if a == 2 and b == 4 or a > 2:
        path = pathsubs + 'tess2018243163601-prf-%d-%d-row%04d-col%04d.fits' % (a, b, indxrows[indxrowsthis], indxcols[indxcolsthis])
    else:
        path = pathsubs + 'tess2018243163600-prf-%d-%d-row%04d-col%04d.fits' % (a, b, indxrows[indxrowsthis], indxcols[indxcolsthis])
    listhdun = astropy.io.fits.open(path)
    listhdun.info()
    psfn = listhdun[0].data
    listhdun.close()
    
    return psfn


def retr_indxtimegoodlyqf(gdat, rflxbkgd):
    
    listindxtimegoodlyqf = [[] for rr in gdat.indxlyqf]
    listindxtimegoodlyqf[0] = np.where(rflxbkgd < 3. * np.median(rflxbkgd))[0]
    listindxtimegoodlyqf[1] = np.where(np.diff(rflxbkgd) < 2.)[0]
    listindxtimegoodlyqf[2] = np.intersect1d(listindxtimegoodlyqf[0], listindxtimegoodlyqf[1])
    
    return listindxtimegoodlyqf


def anim_cntp(gdat, cntp, p, o, typecntpscal, nameplotcntp, \
                  boolresi=False, listindxpixlcolr=None, indxpcol=None, \
                  time=None, lcur=None, \
                  vmin=None, vmax=None, \
                  listtime=None, listtimelabl=None, \
                 ):
    
    if time is None:
        time = [None for t in gdat.indxtime[p][o]]

    if listtimelabl is None:
        listtimelabl = [None for t in gdat.indxtime[p][o]]

    listpath = []
    for tt, t in enumerate(gdat.indxtimeanim):
        
        # make title
        strgtitl = gdat.strgtitlcntpplot
        if listtimelabl[t] is not None:
            strgtitl += ', %s' % listtimelabl[t]
        
        path = plot_cntp(gdat, cntp[:, :, t], p, o, typecntpscal, gdat.pathvisutarg, nameplotcntp, 'fitt', \
                                                strgtitl=strgtitl, boolresi=boolresi, listindxpixlcolr=listindxpixlcolr, \
                                                                                            timelabl=listtimelabl[t], thistime=time[t], indxtimeplot=t, \
                                                                                                vmin=vmin, vmax=vmax, lcur=lcur, time=time)
        
        listpath.append(path)

    return listpath


def plot_histcntp(gdat, cntp, pathvisu, strgvarb):
    
    path = pathvisu + 'hist%s%s.%s' % (strgvarb, gdat.strgsave, gdat.typefileplot)
    if not os.path.exists(path):
        figr, axis = plt.subplots(figsize=gdat.sizefigrsing)
        bins = np.sinh(np.linspace(np.arcsinh(np.nanmin(cntp)), np.arcsinh(np.nanmax(cntp)), 200))
        axis.hist(cntp.flatten(), bins=bins)
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_ylabel('N')
        axis.set_xlabel('C [e$^{-}$/s]')
        plt.tight_layout()
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()


def plot_cntp(gdat, \
              cntp, \
              p, \
              o, \
              typecntpscal, \
              pathbase, \
              nameplotcntp, \
              typecatlplot, \
              indxpcol=None, \
              #cbar='Greys_r', \
              cbar='cividis', \
              strgtitl='', \
              boolresi=False, \
              xposoffs=None, \
              yposoffs=None, \
              
              # type of the scale
              ## 'none': no scale shown
              ## 'angl': angular
              ## 'dist': distance
              typescal='angl', \

              strgextn='', \
              lcur=None, \
              boolanno=True, \
              indxtimeplot=None, \
              time=None, \
              timelabl=None, \
              thistime=None, \
              vmin=None, \
              vmax=None, \
              listindxpixlcolr=None, \
             ):
    
    if typecntpscal == 'asnh':
        cntp = np.arcsinh(cntp)
    
    if strgextn != '':
        strgextn = '_' + strgextn
    
    path = retr_pathvisu(gdat, pathbase, nameplotcntp, typecntpscal=typecntpscal, indxpcol=indxpcol, indxtimeplot=indxtimeplot, typecatlplot=typecatlplot)

    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
    else:
        if boolresi:
            cbar = 'PuOr'
        
        if vmin is None:
            if boolresi:
                vmax = max(np.amax(cntp), abs(np.amin(cntp)))
                vmin = -vmax
            else:
                #vmin = np.amin(cntp[np.where(cntp > 0)])
                vmin = np.amin(cntp)
                vmax = np.amax(cntp)
        
        if lcur is None:
            if gdat.sizeplotcntp == 'singcoln':
                figrsize = gdat.sizefigrsing
            if gdat.sizeplotcntp == 'doubcoln':
                figrsize = gdat.sizefigrdoubimag
            figr, axis = plt.subplots(figsize=figrsize)
            axis =[axis]
        else:
            if gdat.sizeplotcntp == 'singcoln':
                figrsize = gdat.sizefigrsing
            if gdat.sizeplotcntp == 'doubcoln':
                figrsize = gdat.sizefigrdoubimag
            figr, axis = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 0.7]}, figsize=figrsize)
        
        imag = axis[0].imshow(cntp, origin='lower', interpolation='nearest', cmap=cbar, vmin=vmin, vmax=vmax)
        
        # plot the fitting point sources
        if boolanno:
            if typecatlplot == 'refr' or typecatlplot == 'refrfitt':
            
                for q in gdat.refr.indxcatl:
                    # label the reference sources
                    # all reference sources within the field of view
                    axis[0].scatter(gdat.refr.catl[q][p][o]['xpos'][gdat.refr.indxpntswthn[q][p][o]], gdat.refr.catl[q][p][o]['ypos'][gdat.refr.indxpntswthn[q][p][o]], \
                                                                                                        alpha=0.3, s=gdat.sizemrkrsour, color='r', marker='o')
                    # all reference sources within the field of view as well as bright & blended enough
                    axis[0].scatter(gdat.refr.catl[q][p][o]['xpos'][gdat.refr.indxpntswthnbrgt[q][p][o]], gdat.refr.catl[q][p][o]['ypos'][gdat.refr.indxpntswthnbrgt[q][p][o]], \
                                                                                                        alpha=1., s=gdat.sizemrkrsour, color='r', marker='o')
                    # label the reference sources
                    for indxtemp in gdat.refr.indxpntswthn[q][p][o]:
                        axis[0].text(gdat.refr.catl[q][p][o]['xpos'][indxtemp] + 0.5, gdat.refr.catl[q][p][o]['ypos'][indxtemp] + 0.5, \
                                                                                                    gdat.refr.catl[q][p][o]['labl'][indxtemp], color='r')
                    
            if typecatlplot == 'fitt' or typecatlplot == 'refrfitt':
            
                # fitting sources
                xposfitt = np.copy(gdat.fitt.catl['xpos'])
                yposfitt = np.copy(gdat.fitt.catl['ypos'])
                if xposoffs is not None:
                    # add the positional offset, if any
                    xposfitt[0] += xposoffs
                    yposfitt[0] += yposoffs
                ## target
                axis[0].scatter(xposfitt[0], yposfitt[0], alpha=1., color='b', s=gdat.sizemrkrsour, marker='o')
                ## neighbors
                axis[0].scatter(xposfitt[1:], yposfitt[1:], alpha=1., s=gdat.sizemrkrsour, color='b', marker='o')
                for k in gdat.indxpnts:
                    axis[0].text(xposfitt[k] + 0.5, yposfitt[k] + 0.5, '%d' % k, color='b')
        axis[0].set_title(strgtitl) 
        axis[0].set_xticks([])
        axis[0].set_yticks([])
        
        if gdat.numbside[p] < 20:
            # highlight the pixel grid
            for k in range(gdat.numbside[p]+1):
                axis[0].axvline(k - 0.5, ls='--', alpha=0.3, color='y')
            for k in range(gdat.numbside[p]+1):
                axis[0].axhline(k - 0.5, ls='--', alpha=0.3, color='y')
        
        if listindxpixlcolr is not None:
            temp = np.zeros_like(cntp).flatten()
            for indxpixlcolr in listindxpixlcolr:
                rect = patches.Rectangle((indxpixlcolr[1], indxpixlcolr[0]),1,1,linewidth=1,edgecolor='r',facecolor='none')
                axis.add_patch(rect)
        
        if typescal != 'none':
            #xerr = 0.1 * (gdat.numbside[p] - 1)
            #valuarsc = 2 * xerr * gdat.sizepixl[p]
            
            valuarcs = 20.
            xerr = valuarcs / 2. / gdat.sizepixl[p] / gdat.numbside[p]

            lablscal = '%.3g arcsec' % valuarcs
            axis[0].errorbar(0.2, 0.1, xerr=xerr, solid_capstyle='projecting', capsize=5, color='white', transform=axis[0].transAxes)
            axis[0].text(0.2, 0.07, lablscal, color='white', transform=axis[0].transAxes)
    
        #from mpl_toolkits.axes_grid1 import make_axes_locatable    
        #divider = make_axes_locatable(axis[0])
        #cax = divider.append_axes('right', size='5%', pad=0.2)
        #cbar = figr.colorbar(imag, ax=cax)
    
        #axins = inset_axes(ax,
        #                    width="5%",
        #                    height="100%",
        #                    loc='Right',
        #                    #borderpad=-5
        #                   )
        #cbar = figr.colorbar(imag, cax=axins, orientation="horizontal")

        #cbar = figr.colorbar(imag, fraction=0.046, pad=0.04, ax=axis[0]) 
        
        #if typecntpscal == 'asnh':
        #    tick = cbar.ax.get_yticks()
        #    tick = np.sinh(tick)
        #    labl = ['%d' % tick[k] for k in range(len(tick))]
        #    cbar.ax.set_yticklabels(labl)
        
        if lcur is not None:
            axis[1].plot(time - gdat.timeoffs, lcur, color='black', ls='', marker='o', markersize=1)
            axis[1].set_xlabel(gdat.labltime) 
            axis[1].set_ylabel(gdat.lablyaxilcur)
            axis[1].axvline(thistime - gdat.timeoffs)
        
        print('Writing to %s...' % path)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    
    return path


def plot_lcurcomp(gdat, lcurmodl, stdvlcurmodl, k, p, o, strgchun, strgplot, timeedge=None, \
                  strgextn='', \
                  numbcomp=None, \
                  indxtimelimt=None, \
                  indxtzom=None, \
                  boolerrr=False, \
                 ):
    
    timedatatemp = np.copy(gdat.listtime[p][o])
    timerefrtemp = [[] for q in gdat.refr.indxtser[p][o]] 
    for q in gdat.refr.indxtser[p][o]:
        timerefrtemp[q] = gdat.refr.time[p][o][q]
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    nameplot = 'tser%scomp_%s_Component%02d' % (gdat.strgnorm, strgplot, k)
    path = retr_pathvisu(gdat, gdat.pathvisutarg, nameplot, indxtzom=indxtzom)
    
    # skip the plot if it has been made before
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
        return

    figr, axis = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [2, 0.7]}, figsize=gdat.sizefigrdoublcur)
    axis[1].set_xlabel(gdat.labltime)
    
    axis[0].set_ylabel(gdat.lablyaxilcur)
    axis[1].set_ylabel('Residual')
    
    if boolerrr:
        yerr = stdvlcurmodl
    else:
        yerr = None
    temp, listcaps, temp = axis[0].errorbar(timedatatemp - gdat.timeoffs, lcurmodl, yerr=yerr, color='gray', ls='', ms=1, \
                                                                                marker='.', lw=3, alpha=0.5, label='Lygos', rasterized=True)
    for caps in listcaps:
        caps.set_markeredgewidth(3)
    
    if timeedge is not None:
        for timeedgetemp in timeedge[1:-1]:
            axis[0].axvline(timeedgetemp - gdat.timeoffs, ls='--', color='gray', alpha=0.5)
    
    for q in gdat.refr.indxtser[p][o]:
        if boolerrr:
            yerr = gdat.refr.stdvrflx[p][o][q]
        else:
            yerr = None
        print('q')
        print(q)
        print('gdat.refr.indxtser[p][o]')
        print(gdat.refr.indxtser[p][o])
        print('gdat.refr.colrlcur')
        print(gdat.refr.colrlcur)
        print(' gdat.refr.rflx')
        print( gdat.refr.rflx)
        print('gdat.refr.labltser')
        print(gdat.refr.labltser)
        print('timerefrtemp[q]')
        print(timerefrtemp[q])
        print('gdat.refr.labltser[p][o][q]')
        print(gdat.refr.labltser[p][o][q])
        print('gdat.refr.rflx[p][o][q]')
        print(gdat.refr.rflx[p][o][q])
        temp, listcaps, temp = axis[0].errorbar(timerefrtemp[q] - gdat.timeoffs, gdat.refr.rflx[p][o][q], \
                                            yerr=yerr, color=gdat.refr.colrlcur[q], ls='', markersize=2, \
                                                                    marker='.', lw=3, alpha=0.3, label=gdat.refr.labltser[p][o][q])
        for caps in listcaps:
            caps.set_markeredgewidth(3)
    
    ## residual
    for q in gdat.refr.indxtser[p][o]:
        if lcurmodl.size == gdat.refr.rflx[p][o][q].size:
            ydat = lcurmodl - gdat.refr.rflx[p][o][q]
            if boolerrr:
                yerr = None
            else:
                yerr = None
            axis[1].errorbar(timedatatemp - gdat.timeoffs, ydat, yerr=yerr, label=gdat.refr.labltser[p][o][q], \
                                                color='k', ls='', marker='.', markersize=2, alpha=0.3)
    
    
    strgtitl = '%s, %s, %s' % (gdat.fitt.lablcomp[k], gdat.liststrginst[p], gdat.listlablpoin[p][o])
    axis[0].set_title(strgtitl)
    if gdat.listtimeplotline is not None:
        for timeplotline in gdat.listtimeplotline:
            axis[0].axvline(timeplotline - gdat.timeoffs, ls='--')
    
    if gdat.refr.numbtser[p][o] > 0:
        axis[0].legend()

    if indxtzom is not None:
        axis[a].set_xlim(gdat.listlimttimetzom[indxtzom] - gdat.timeoffs)
    
    #plt.tight_layout()
    print('Writing to %s...' % path)
    plt.savefig(path, dpi=200)
    plt.close()


    
def plot_lcur(gdat, lcurmodl, stdvlcurmodl, k, p, o, strgchun, strgplot, cdpp=None, \
                    timeedge=None, \
                    numbcomp=None, \
                    strgextn='', indxtimelimt=None, indxtzom=None, boolerrr=False):
    
    timedatatemp = np.copy(gdat.listtime[p][o])
    
    if indxtimelimt is not None:
        timedatatemp = timedatatemp[indxtimelimt]
        stdvlcurmodl = stdvlcurmodl[indxtimelimt]
        lcurmodl = lcurmodl[indxtimelimt]

    nameplot = 'tser%s_%s_Component%02d' % (gdat.strgnorm, strgplot, k)
    path = retr_pathvisu(gdat, gdat.pathvisutarg, nameplot, indxtzom=indxtzom)
    
    # skip the plot if it has been made before
    if os.path.exists(path):
        print('Plot exists at %s. Skipping the plotting...' % path)
        return

    figr, axis = plt.subplots(figsize=gdat.sizefigrdoublcur)
    axis.set_xlabel(gdat.labltime)
    
    axis.set_ylabel(gdat.lablyaxilcur)
    
    if boolerrr:
        yerr = stdvlcurmodl
    else:
        yerr = None
    temp, listcaps, temp = axis.errorbar(timedatatemp - gdat.timeoffs, lcurmodl, yerr=yerr, color='gray', ls='', ms=1, marker='.', lw=3, rasterized=True)
    for caps in listcaps:
        caps.set_markeredgewidth(3)
    
    if timeedge is not None:
        for timeedgetemp in timeedge[1:-1]:
            axis.axvline(timeedgetemp - gdat.timeoffs, ls='--', color='gray', alpha=0.5)
    
    # make the title
    strgtitl = '%s, %s, %s' % (gdat.fitt.lablcomp[k], gdat.liststrginst[p], gdat.listlablpoin[p][o])
    if not cdpp is None:
        strgtitl += ', 1-hr CDPP = %.1f ppm' % cdpp
    
    axis.set_title(strgtitl)
    if gdat.listtimeplotline is not None:
        for timeplotline in gdat.listtimeplotline:
            axis.axvline(timeplotline - gdat.timeoffs, ls='--')
    
    if indxtzom is not None:
        axis.set_xlim(gdat.listlimttimetzom[indxtzom] - gdat.timeoffs)
    
    plt.subplots_adjust(bottom=0.2)
    print('Writing to %s...' % path)
    plt.savefig(path, dpi=200)
    plt.close()


def retr_pathvisu(gdat, \
                  pathbase, \
                  nameplot, \

                  # rflx 
                  # temporal zoom
                  indxtzom=None, \
                    
                  typecatlplot=None, \

                  # cntp
                  typecntpscal=None, \
                  indxpcol=None, \
                  typevisu='plot', \
                  indxtimeplot=None, \
                 ):
    
    if typecntpscal is None or typecntpscal == 'self':
        strgscal = ''
    else:
        strgscal = '_' + typecntpscal

    if indxtzom is None:
        strgtzom = ''
    else:
        strgtzom = '_zom%d' % indxtzom
    
    if typecatlplot is None:
        strgtypecatlplot = ''
    else:
        strgtypecatlplot = '_catl' + typecatlplot

    if indxpcol is None:
        strgpcol = ''
    else:
        strgpcol = '_col%d' % indxpcol
    
    if typevisu == 'anim':
        typefileplot = 'gif'
    else:
        typefileplot = gdat.typefileplot
    
    if indxtimeplot is None:
        strgtime = ''
    else:
        strgtime = '_%08d' % indxtimeplot
    
    pathvisu = pathbase + '%s_%s%s%s%s%s%s.%s' % (nameplot, gdat.strgsave, strgscal, strgtzom, strgpcol, strgtime, strgtypecatlplot, typefileplot)
    
    return pathvisu


def retr_cntpmodl(gdat, p, strgmodl, xpos, ypos, cnts, cntpbackscal, parapsfn, typesour):
    '''
    Calculate the model image
    '''
    
    gmod = getattr(gdat, strgmodl)
    
    if cnts.ndim == 2:
        cntpmodl = np.zeros((gdat.numbside[p], gdat.numbside[p], cnts.shape[0])) + cntpbackscal[None, None, :]
    else:
        cntpmodl = np.zeros((gdat.numbside[p], gdat.numbside[p])) + cntpbackscal
                
    if typesour == 'pnts':
        if gdat.typepsfnsubp == 'eval':
             
            for k in range(xpos.size):
                
                # temp check if -1 is correct
                arryxpos = np.arange(int(xpos[k]) - gdat.numbsideevalhalf, int(xpos[k]) + gdat.numbsideevalhalf - 1)
                arryypos = np.arange(int(ypos[k]) - gdat.numbsideevalhalf, int(ypos[k]) + gdat.numbsideevalhalf - 1)
                arryxposmesh, arryyposmesh = np.meshgrid(arryxpos, arryypos)
                arryxposmesh = arryxposmesh.flatten()
                arryyposmesh = arryyposmesh.flatten()

                deltxpos = gdat.xposimag[p][arryxposmesh, arryyposmesh] - xpos[k]
                deltypos = gdat.yposimag[p][arryxposmesh, arryyposmesh] - ypos[k]

                if gmod.typepsfnshap == 'gauscirc':
                    psfnsour = np.exp(-(deltxpos / parapsfn[0])**2 - (deltypos / parapsfn[0])**2)
                if gmod.typepsfnshap.startswith('gauselli'):
                    gausbvar = np.exp(-0.5 * (deltxpos / parapsfn[0])**2 - 0.5 * (deltypos / parapsfn[1])**2)
                    psfnsour = gausbvar + parapsfn[2] * deltxpos * gausbvar + parapsfn[3] * deltypos * gausbvar
                    psfnsour[psfnsour < 0] = 0.
                if gmod.typepsfnshap == 'empi':
                    z = parapsfn.reshape((gdat.numbsidepsfnfitt, gdat.numbsidepsfnfitt))
                    psfnsour = scipy.interpolate.interp2d(gdat.xposimagpsfn[p], gdat.yposimagpsfn[p], z, kind='cubic', fill_value=0.)(gdat.xposimag[p], gdat.yposimag[p])

                if gmod.typepsfnshap == 'pfre':
                    psfnsour = coef[0] * deltxpos + coef[1] * deltypos + coef[2] * deltxpos * deltypos + \
                                    coef[3] * deltxpos**2 + coef[4] * deltypos**2 + coef[5] * deltxpos**2 * deltypos + coef[6] * deltypos**2 * deltxpos + \
                                    coef[7] * deltxpos**3 + coef[8] * deltypos**3 + coef[9] * np.exp(-deltxpos**2 / coef[10] - deltypos**2 / coef[11])
                
                if cnts.ndim == 2:
                    cntpmodl[arryxposmesh, arryyposmesh] += cnts[None, None, :, k] * psfnsour[:, :, None]
                else:
                    cntpmodl[arryxposmesh, arryyposmesh] += cnts[k] * psfnsour
        
        if gdat.typepsfnsubp == 'regrcubipixl':
             
            # construct the design matrix
            ## subpixel shifts
            dx = xpos - xpos.astype(int) - 0.5
            dy = ypos - ypos.astype(int) - 0.5
            ## design matrix
            matrdesi = np.column_stack((np.ones(x.size), dx, dy, dx*dx, dx*dy, dy*dy, dx*dx*dx, dx*dx*dy, dx*dy*dy, dy*dy*dy))
            
            tmpt = np.dot(matrdesi, gdat.coefspix).reshape((-1, gdat.numbside[p], gdat.numbside[p]))
            cntpmodl = np.sum(tmpt[:, :, :, None] * cnts[:, None, None, :], 0)
            
            if gdat.booldiag:
                if not np.isfinite(cntpmodl).all():
                    print('WARNING: cntpmodl was infinite!!!!')
                    raise Exception('')
    
    if typesour == 'Earth':
        cntpmodl += gdat.funcepic * cnts

    return cntpmodl


def retr_dictderipsfn(para, gdat):
    
    dictvarbderi = dict()
    
    if gdat.typefittpsfnposi == 'fixd':
        xpos = gdat.fitt.catl['xpos']
        ypos = gdat.fitt.catl['ypos']
    if gdat.typefittpsfnposi == 'vari':
        xpos = para[gdat.indxparapsfnxpos] + gdat.fitt.catl['xpos']
        ypos = para[gdat.indxparapsfnypos] + gdat.fitt.catl['ypos']
    
    cnts = para[gdat.indxparapsfnflux]
    cntpbackscal = para[gdat.indxparapsfnback]
    parapsfn = para[gdat.indxparapsfnpsfn]
    
    cntpmodl = retr_cntpmodl(gdat, p, 'fitt', xpos, ypos, cnts, cntpbackscal, parapsfn, 'pnts')
    dictvarbderi['cntpmodlsexp'] = cntpmodl
    
    cntpmodlpnts = retr_cntpmodl(gdat, p, 'fitt', xpos, ypos, cnts, 0., parapsfn, 'pnts')
    dictvarbderi['cntpmodlpntssexp'] = cntpmodlpnts
    
    dictvarbderi['cntpresisexp'] = gdat.cntpdatasexp - cntpmodl
    
    if xpos.size > 1:
        dictvarbderi['raticont'] = retr_raticonttotl(gdat, p, xpos, ypos, cnts, parapsfn)

    if xpos[0] == (gdat.numbside[p] - 1.) / 2. and ypos[0] == (gdat.numbside[p] - 1.) / 2. and xpos.size == 1:
        intg = int((gdat.numbside[p] - 1.) / 2.)
        dictvarbderi['fraccent'] = cntpmodlpnts[intg, intg] / np.sum(cntpmodlpnts)

    return dictvarbderi


def retr_raticontsing(gdat, p, xpos, ypos, cnts, parapsfn):
    
    cntpmodl = np.empty((gdat.numbside[p], gdat.numbside[p], xpos.size))
    for k in range(len(xpos)):
        cntpmodl[:, :, k] = retr_cntpmodl(gdat, p, 'fitt', xpos[k, None], ypos[k, None], cnts[k, None], 0., parapsfn, 'pnts')
    
    # contamination ratio
    raticont = np.empty(xpos.size - 1)
    for k in range(len(xpos)-1):
        raticont[k] = np.sum(cntpmodl[:, :, 0] * cntpmodl[:, :, k+1])
    raticont /= np.sum(cntpmodl[:, :, 0])**2
    
    return raticont


def retr_raticonttotl(gdat, p, xpos, ypos, cnts, parapsfn):
    
    raticontsing = retr_raticontsing(gdat, p, xpos, ypos, cnts, parapsfn)
    raticonttotl = np.sum(raticontsing)

    #cntpmodltarg = retr_raticont(gdat, xpos[0, None], ypos[0, None], cnts[0, None], parapsfn)
    #cntpmodlneig = retr_raticont(gdat, xpos[1:], ypos[1:], cnts[1:], parapsfn)
    #raticont = np.sum(cntpmodltarg * cntpmodlneig) / np.sum(cntpmodltarg)**2
    
    return raticonttotl


def retr_llik(para, gdat):
    
    # parse the parameter vector
    if gdat.typefittpsfnposi == 'fixd':
        xpos = gdat.fitt.catl['xpos']
        ypos = gdat.fitt.catl['ypos']
    if gdat.typefittpsfnposi == 'vari':
        xpos = para[:gdat.fitt.numbpnts]
        ypos = para[gdat.fitt.numbpnts:2*gdat.fitt.numbpnts]
    
    cnts = para[gdat.indxparapsfnflux]
    cntpbackscal = para[gdat.indxparapsfnback]
    parapsfn = para[gdat.indxparapsfnpsfn]
    
    cntpmodl = retr_cntpmodl(gdat, p, 'fitt', xpos, ypos, cnts, cntpbackscal, parapsfn, gdat.typesour)
    
    chi2 = np.sum((gdat.cntpdatasexp - cntpmodl)**2 / gdat.cntpdatasexp)
    
    llik = -0.5 * chi2
    
    #print('xpos')
    #print(xpos)
    #print('ypos')
    #print(ypos)
    #print('cnts')
    #print(cnts)
    #print('cntpbackscal')
    #print(cntpbackscal)
    #print('parapsfn')
    #print(parapsfn)
    #print('gdat.cntpdatatmed')
    #print(gdat.cntpdatatmed)
    #print('cntpmodl')
    #print(cntpmodl)
    
    if gdat.booldiag and not np.isfinite(llik):
        print('')
        print('')
        print('')
        print('gdat.cntpdatasexp')
        summgene(gdat.cntpdatasexp)
        print('xpos')
        summgene(xpos)
        print('ypos')
        summgene(ypos)
        print('cnts')
        summgene(cnts)
        print('cntpbackscal')
        summgene(cntpbackscal)
        print('parapsfn')
        summgene(parapsfn)
        print('cntpmodl')
        summgene(cntpmodl)
        print('llik')
        print(llik)
        raise Exception('Likelihood is Infinite!')

    return llik


#@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def retr_mlikregr(cntpdataflat, matrdesi, variflat):
    '''
    Return the maximum likelihood estimate for linear regression.
    '''
    
    varitdim = np.diag(variflat)
    covafittcnts = np.linalg.inv(np.matmul(np.matmul(matrdesi.T, np.linalg.inv(varitdim)), matrdesi))
    mlikfittcnts = np.matmul(np.matmul(np.matmul(covafittcnts, matrdesi.T), np.linalg.inv(varitdim)), cntpdataflat)
    
    return mlikfittcnts, covafittcnts


def updt_strgsave(gdat, strgchun, x, y, p, o):
    
    strgnumbside = '_n%03d' % gdat.numbside[p]
    strgmaxmdmag = '_d%3.1f' % gdat.maxmdmag
    strgoffs = '_of%d%d' % (x, y)
    gdat.strgsave = '%s_%s_%s_%s%s%s%s' % (gdat.strgcnfg, gdat.liststrginst[p], strgchun, gdat.typecade[p][o], strgnumbside, strgmaxmdmag, strgoffs)


def setp_cntp(gdat, strg, typecntpscal):
    
    cntp = getattr(gdat, strg)
    
    if strg.startswith('resi'):
        vmin = np.nanpercentile(cntp, 0)
        vmax = np.nanpercentile(cntp, 100)
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    else:
        vmin = np.nanpercentile(cntp[np.where(cntp > 0.)], 0)
        vmax = np.nanpercentile(cntp, 100)
    
    if typecntpscal == 'asnh':
        vmin = np.arcsinh(vmin)
        vmax = np.arcsinh(vmax)
    setattr(gdat, 'vmin' + strg + typecntpscal, vmin)
    setattr(gdat, 'vmax' + strg + typecntpscal, vmax)
    

def retr_strgchun(gdat, p, o):
    
    if gdat.liststrginst[p] == 'TESS':
        strgchun = '%02d%d%d' % (gdat.listipnt[p][o], gdat.listtcam[o], gdat.listtccd[o])
    else:
        strgchun = 'ch%02d' % gdat.listipnt[p][o]
        
    return strgchun


def init( \

         # data
         ## type of data:
         ## 'simutargsynt': simulated data on a synthetic target
         ## 'simutargpartsynt': simulated synthetic data on a particular target with a particular observational baseline 
         ## 'simutargpartinje': simulated data obtained by injecting a synthetic signal on observed data on a particular target with a particular observational baseline 
         ## 'obsd': observed data on a particular target
         liststrgtypedata=None, \
         
         # selected TESS sectors
         listtsecsele=None, \
    
         # selected pointings
         listipntsele=None, \
    
         # data
         ## number of pixels on a side to cut out
         numbside=None, \
         
         ## Boolean flag to use Target Pixel Files (TPFs) at the highest cadence whenever possible
         boolutiltpxf=True, \
         
         ## target
         ### a string to be used to search MAST for the target
         strgmast=None, \
         ### TIC ID of the target
         ticitarg=None, \
         ### TOI ID
         toiitarg=None, \
         ### RA of the target
         rasctarg=None, \
         ### DEC of the target
         decltarg=None, \
         ### name of simulated object (reserved for simulated objects not on MAST, not in TCI, not a TOI or not at particular RA & DEC)
         nametarg=None, \
         ### estimated target properties if RA and DEC are provided as target
         #### estimated number of counts
         cntsestitarg=None, \
         #### TESS magntidue
         tmagtarg=None, \
         #### proper motion along RA
         pmratarg=None, \
         #### proper motion along RA
         pmdetarg=None, \
         #### original RA (before proper motion correction)
         rascorigtarg=None, \
         #### original Dec (before proper motion correction)
         declorigtarg=None, \

         # type of the instrument
         liststrginst=None, \

         # simulated data
         ### time array for simulation
         listtime=None, \
         ## a dictionary describing the true generative model
         ### numbtime: number of time bins
         ### sourtype: 'dwar', \
         ### xpostarg: None, \
         ### ypostarg: None, \
         ### tmagtarg: None, \
         ### pmratarg: None, \
         ### pmdetarg: None, \
         ### pmxatarg: None, \
         ### pmyatarg: None, \
         dicttrue=None, \
        
         seedrand=0, \

         ## mask
         ### Boolean flag to put a cut on quality flag
         boolmaskqual=True, \
         ### limits of time between which the quality mask will be ignored
         limttimeignoqual=None, \
         ### ephemeris mask
         epocmask=None, \
         perimask=None, \
         duramask=None, \
        
         # a string that will cause the run output to be placed in a subfolder inside the folder for the target, where the run output will be placed
         strgruns=None, \

         # list of photometric analyses on the image
         listnameanls=None, \

         #Boolean flag to normalize the light curve by the median
         boolnorm=None, \

         # type of normaliztion
         ## 'medi': divide by median
         ## 'mediinit': divide by median in the first 1/10th
         typenorm='medi', \

         # Boolean flag to merge fitting sources that are too close
         boolmerg=True, \

         # processing
         ## Boolean flag to turn on CBV detrending
         booldetrcbvs=False, \
         
         # exposure times for each instrument and pointing
         timeexpo=None, \

         # string indicating the cluster of targets
         strgclus=None, \
         
         # reference time-series
         refrarrytser=None, \
         # list of labels for reference time-series
         refrlistlabltser=None, \

         # maximum delta magnitude of neighbor sources to be included in the model
         maxmdmag=4., \
         
         # maximum TESS magnitude of the objects to be queried on MAST
         maxmtmagcatl=None, \

         # model
         ## Boolean flag to perform the photometric fit
         boolfitt=True, \

         # dictionary for parameters of the fitting generative model
         dictfitt=dict(), \

         ### RA of the target
         #fittrasctarg=None, \
         ### DEC of the target
         #fittdecltarg=None, \
         ## horizontal position of the target
         #fittxpostarg=None, \
         ## vertical position of the target
         #fittypostarg=None, \
         ## TESS magniude of the target
         #fitttmagtarg=None, \

         #fittxposneig=None, \
         #fittyposneig=None, \
         #fitttmagneig=None, \
        
         ### RA proper motion, used when RA and DEC are provided by the user
         #fittpmratarg=None, \
         ### DEC proper motion, used when RA and DEC are provided by the user
         #fittpmdetarg=None, \
         
         ## PSF
         ### type of inference
         #### 'fixd': fixed
         #### 'locl': based on only the image data to be analyzed
         #### 'osam': based on the dithered image data collected during commissioning
         #### 'osam': based on only the dithered image data collected during commissioning
         #### 'both': based on both
         typepsfninfe='fixd', \
         
         ### type of sub-pixel interpolation of the point source emission model
         #### 'eval': evaluation of a functional form on the data grid (not to be used when point source positions are floating)
         #### 'regrcubi': cubic regression
         #### 'regrline': linear regression
         #### 'regrcubipixl': separate cubic regression in each pixel
         typepsfnsubp='eval', \
         
         catlextr=None, \
         lablcatlextr=None, \
            
         # Boolean flag to repeat the fit, putting the target to offset locations
         boolfittoffs=False, \
         
         # Maximum radius for query [arcsec]
         maxmradiquer=None, \

         # Boolean flag to use fast cadence (20 sec) when using target pixel files
         boolfasttpxf=False, \

         ## post-process
         ### Boolean flag to perform aperture photometry
         boolphotaper=True, \
         ### list of pixels for the aperture
         listpixlapertarg=None, \
         ### Boolean flag to output aperture photometry
         booloutpaper=False, \

         # visualization
         ## Boolean flag to make any plots
         boolplot=True, \
         ## Boolean flag to make relative flux plots
         boolplotrflx=None, \
         ## Boolean flag to make image plots
         boolplotcntp=None, \
         ## Boolean flag to plot the centroiod
         boolplotcent=None, \
         ## Boolean flag to plot the quaternions
         boolplotquat=None, \
         ## Boolean flag to plot the histogram of the number of counts
         boolplothhistcntp=None, \
         ## Boolean flag to make an animation
         boolanim=None, \
         ## Boolean flag to include all time bins in the animation
         boolanimframtotl=None, \
        
         # string indicating the size of the figures showing maps
         sizeplotcntp='doubcoln', \

         ## time offset for time-series plots
         timeoffs=2457000., \

         ## list of limits for temporal zoom
         listlimttimetzom=None, \

         ## the time to indicate on the plots with a vertical line
         listtimeplotline=None, \
         
         # path for the target
         pathtarg=None, \
        
         # factor to scale the size of text in the figures
         factsizetextfigr=1., \

         # a string that will be used to name the folder for this target
         strgtarg=None, \
         
         # a string that will appear in the plots to label the target, which can be anything the user wants
         labltarg=None, \
         
         # image color scale
         ## 'self': image is linear
         ## 'asnh': arcsinh of the image is linear
         listtypecntpscal=['asnh'], \
        
         # plot extensions
         typefileplot='png', \
        
         # type of verbosity
         ## -1: absolutely no text
         ##  0: no text output except critical warnings
         ##  1: minimal description of the execution
         ##  2: detailed description of the execution
         typeverb=1, \
        
         # Boolean flag to turn on diagnostic mode
         booldiag=True, \
         
        ):
    '''
    Perform photometry on time-series imaging data from TESS or LSST
    '''
    
    # start the timer
    timeinittotl = timemodu.time()
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)
    
    if gdat.boolnorm is None:
        gdat.boolnorm = True
    
    if gdat.boolnorm:
        gdat.strgnorm = 'rflx'
    else:
        # ADC counts per Second
        gdat.strgnorm = 'adcs'

    if gdat.boolnorm:
        gdat.lablyaxilcur = 'Relative flux'
    else:
        gdat.lablyaxilcur = 'ADC counts [e$^-$/s]'
    
    # copy named arguments to the global object
    #for strg, valu in args.items():
    #    setattr(gdat, strg, valu)
    
    # set the seed
    np.random.seed(seedrand)
    
    if gdat.boolplot:
        
        if gdat.boolplotrflx is None:
            gdat.boolplotrflx = True
        
        if gdat.boolplotcntp is None:
            gdat.boolplotcntp = True
         
        if gdat.boolplotquat is None:
            gdat.boolplotquat = True
         
        if gdat.boolanim is None:
            gdat.boolanim = False
         
        if gdat.boolanimframtotl is None:
            gdat.boolanimframtotl = False
        
        if gdat.boolplothhistcntp is None:
            gdat.boolplothhistcntp = False

    if isinstance(gdat.liststrginst, list):
        gdat.liststrginst = np.array(gdat.liststrginst)
    
    gdat.numbinst = len(gdat.liststrginst)
    gdat.indxinst = np.arange(gdat.numbinst)
    
    if gdat.booldiag:
        for p in gdat.indxinst:
            if not gdat.liststrgtypedata[p] in ['simutargsynt', 'simutargpartsynt', 'simutargpartinje', 'obsd']:
                print('')
                print('')
                print('')
                print('gdat.liststrgtypedata')
                print(gdat.liststrgtypedata)
                raise Exception('gdat.liststrgtypedata has an issue.')

    if gdat.numbside is None:
        gdat.boolinptnumbside = False
        gdat.numbsidedefa = 11
        gdat.numbside = np.full(gdat.numbinst, gdat.numbsidedefa)
    else:
        gdat.boolinptnumbside = True
        
    if isinstance(gdat.numbside, list):
        gdat.numbside = np.array(gdat.numbside)

    if gdat.dicttrue is None:
        gdat.dicttrue = dict()
    
    if gdat.maxmtmagcatl is None:
        gdat.maxmtmagcatl = 20.

    if gdat.booldiag:
        for p in gdat.indxinst:
            if gdat.liststrgtypedata[p] == 'simutargpartinje':
                if 'rasctarg' in gdat.true or 'decltarg' in gdat.true:
                    raise Exception('When a sim. source is injected, only provide target RA and Dec and exclude them from the generative model dictionary (dicttrue).')
    
    gdat.true = tdpy.util.gdatstrt()
    for p in gdat.indxinst:
        if gdat.liststrgtypedata[p].startswith('simu'):
            for name, valu in gdat.dicttrue.items():
                setattr(gdat.true, name, valu)

    # parameters for the true model (also used as default parameters for the fitting model)
    ## right ascension of the target
    tdpy.setp_para_defa(gdat, 'true', 'rasctarg', gdat.rasctarg)
    
    ## declination of the target
    tdpy.setp_para_defa(gdat, 'true', 'decltarg', gdat.decltarg)
    
    ## proper motion along the horizontal axis
    tdpy.setp_para_defa(gdat, 'true', 'velxtarg', 0.)
    
    ## proper motion along the vertical axis
    tdpy.setp_para_defa(gdat, 'true', 'velytarg', 0.)
    
    ## Boolean flag indicating if the target is a Solar System object
    tdpy.setp_para_defa(gdat, 'true', 'booltargssob', False)
    
    ## proper motion in RA
    tdpy.setp_para_defa(gdat, 'true', 'velrtarg', 0.)
    
    ## proper motion in declination
    tdpy.setp_para_defa(gdat, 'true', 'veldtarg', 0.)
    
    ## type of PSF model shape
    tdpy.setp_para_defa(gdat, 'true', 'typepsfnshap', 'gauscirc')

    ## full width at half maximum of the point-spread function
    tdpy.setp_para_defa(gdat, 'true', 'sigmpsfn', 1.05)
    tdpy.setp_para_defa(gdat, 'true', 'sigmpsfnxpos', 1.05)
    tdpy.setp_para_defa(gdat, 'true', 'sigmpsfnypos', 1.05)

    tdpy.setp_para_defa(gdat, 'true', 'fracskewpsfnxpos', 0.)
    tdpy.setp_para_defa(gdat, 'true', 'fracskewpsfnypos', 0.)
    tdpy.setp_para_defa(gdat, 'true', 'cntpbackscal', 100.)
    
    gdat.maxmnumbside = np.amax(gdat.numbside)

    tdpy.setp_para_defa(gdat, 'true', 'xpostarg', (gdat.maxmnumbside - 1.) / 2.)
    tdpy.setp_para_defa(gdat, 'true', 'ypostarg', (gdat.maxmnumbside - 1.) / 2.)
    tdpy.setp_para_defa(gdat, 'true', 'tmagtarg', 10.)
    tdpy.setp_para_defa(gdat, 'true', 'xposneig', np.array([]))
    tdpy.setp_para_defa(gdat, 'true', 'xposneig', np.array([]))
    tdpy.setp_para_defa(gdat, 'true', 'tmagneig', np.array([]))
    
    gdat.numbsideevalhalf = 5
    
    gdat.fitt = tdpy.util.gdatstrt()
    if gdat.dictfitt is not None:
        for name, valu in gdat.dictfitt.items():
            setattr(gdat.fitt, name, valu)
    
    # assign true model parameters to the default fitting model parameters
    for name in ['typepsfnshap', 'sigmpsfn', 'sigmpsfnxpos', 'sigmpsfnypos', 'fracskewpsfnxpos', 'fracskewpsfnypos']:
        if not hasattr(gdat.fitt, name):
            setattr(gdat.fitt, name, getattr(gdat.true, name))
    
    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('')
    print('')
    print('')
    print('lygos initialized at %s...' % gdat.strgtimestmp)
    # paths
    gdat.pathbase = os.environ['LYGOS_DATA_PATH'] + '/'
    gdat.pathvisulygo = gdat.pathbase + 'visuals/'
    gdat.pathdatalygo = gdat.pathbase + 'data/'
    
    np.set_printoptions(linewidth=200, \
                        precision=5, \
                       )
    
    print('gdat.liststrgtypedata')
    print(gdat.liststrgtypedata)
    
    # check input
    ## ensure target identifiers are not conflicting
    if gdat.booldiag:
        for p in gdat.indxinst:
            if gdat.liststrgtypedata[p] == 'simutargpartinje' or gdat.liststrgtypedata[p] == 'obsd':
                if not (gdat.ticitarg is not None and gdat.strgmast is None and gdat.toiitarg is None and gdat.rasctarg is None and gdat.decltarg is None or \
                    gdat.ticitarg is None and gdat.strgmast is not None and gdat.toiitarg is None and gdat.rasctarg is None and gdat.decltarg is None or \
                    gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is not None and gdat.rasctarg is None and gdat.decltarg is None or \
                    gdat.ticitarg is None and gdat.strgmast is None and gdat.toiitarg is None and gdat.rasctarg is not None and gdat.decltarg is not None):
                    print('')
                    print('')
                    print('')
                    print('gdat.ticitarg')
                    print(gdat.ticitarg)
                    print('gdat.strgmast')
                    print(gdat.strgmast)
                    print('gdat.toiitarg')
                    print(gdat.toiitarg)
                    print('gdat.rasctarg')
                    print(gdat.rasctarg)
                    print('gdat.decltarg')
                    print(gdat.decltarg)
                    raise Exception('Either a TIC ID (ticitarg), RA&DEC (rasctarg and decltarg), MAST key (strgmast) or a TOI number (toiitarg) should be provided.')
        
            if gdat.liststrgtypedata[p] == 'simutargsynt' and gdat.true.tmagtarg is None:
                print('')
                print('')
                print('')
                print('gdat.true.tmagtarg')
                print(gdat.true.tmagtarg)
                raise Exception('truetmagtarg needs to be set when simulated data based on a generative model, are generated.')

    if gdat.ticitarg is not None or gdat.toiitarg is not None:
        dicttoii = nicomedia.retr_dicttoii()

    # settings
    ## plotting

    # conversion factors
    gdat.dictfact = tdpy.retr_factconv()
    
    # types of ticitarg
    #'TICID', 'TOIID', 'mast', 'posi', 'sols', 'posi'

    # determine the MAST keyword and TOI ID of the target and its type
    if gdat.ticitarg is not None:
        gdat.typetarg = 'TICID'
        print('A TIC ID was provided as target identifier.')
        indx = np.where(dicttoii['TICID'] == gdat.ticitarg)[0]
        if indx.size > 0:
            gdat.toiitarg = int(str(dicttoii['TOIID'][indx[0]]).split('.')[0])
            print('Matched the input TIC ID with TOI %d.' % gdat.toiitarg)
        gdat.strgmast = 'TIC %d' % gdat.ticitarg
    elif gdat.toiitarg is not None:
        gdat.typetarg = 'TOIID'
        print('A TOI number (%d) was provided as target identifier.' % gdat.toiitarg)
        # determine TIC ID
        gdat.strgtoiibase = str(gdat.toiitarg)
        indx = []
        for k, strg in enumerate(dicttoii['TOIID']):
            if str(strg).split('.')[0] == gdat.strgtoiibase:
                indx.append(k)
        indx = np.array(indx)
        if indx.size == 0:
            print('Did not find the TOI in the ExoFOP-TESS TOI list.')
            print('dicttoii[toii]')
            summgene(dicttoii['TOIID'])
            raise Exception('')
        gdat.ticitarg = dicttoii['TICID'][indx[0]]
        gdat.strgmast = 'TIC %d' % gdat.ticitarg
    elif gdat.strgmast is not None:
        gdat.typetarg = 'mast'
        print('A MAST key (%s) was provided as target identifier.' % gdat.strgmast)
    elif gdat.rasctarg is not None and gdat.decltarg is not None:
        gdat.typetarg = 'posi'
        print('RA and DEC (%g %g) are provided as target identifier.' % (gdat.rasctarg, gdat.decltarg))
        gdat.strgmast = '%g %g' % (gdat.rasctarg, gdat.decltarg)
    elif gdat.nametarg is not None:
        gdat.typetarg = 'sols'
        print('A Solar System object is provided as target identifier: %s.' % gdat.nametarg)
        print('temp: Making up an RA and DEC for the target...')
        gdat.rasctarg = 0.
        gdat.decltarg = 0.
    elif gdat.true.xpostarg is not None and gdat.true.ypostarg is not None:
        gdat.typetarg = 'posi'
        print('Horizontal and vertical positions (%g %g) are provided as target identifier for a simulated target.' % (gdat.true.xpostarg, gdat.true.ypostarg))
    else:
        raise Exception('')

    print('gdat.typetarg')
    print(gdat.typetarg)
    booltemp = False
    for p in gdat.indxinst:
        if gdat.liststrgtypedata[p] == 'simutargpartinje' or gdat.liststrgtypedata[p] == 'obsd':
            booltemp = True
    
    if booltemp:
        print('gdat.strgmast')
        print(gdat.strgmast)
        print('gdat.toiitarg')
        print(gdat.toiitarg)
    
    # pixel size in arcseconds
    gdat.sizepixl = np.empty(gdat.numbinst)
    for p in gdat.indxinst:
        if gdat.liststrginst[p].startswith('LSST'):
            gdat.sizepixl[p] = 0.2
        elif gdat.liststrginst[p].startswith('TESS') or gdat.liststrginst[p] == 'TGEO-IR':
            # temp
            gdat.sizepixl[p] = 21.
        elif gdat.liststrginst[p] == 'TESSCam':
            # temp
            gdat.sizepixl[p] = 21.
        elif gdat.liststrginst[p] == 'ULTRASAT':
            gdat.sizepixl[p] = 2.
        elif gdat.liststrginst[p] == 'TGEO-UV':
            gdat.sizepixl[p] = 0.2
        elif gdat.liststrginst[p] == 'TGEO-VIS':
            gdat.sizepixl[p] = 0.2
        else:
            print('gdat.liststrginst')
            print(gdat.liststrginst)
            raise Exception('')
    
    if gdat.booldiag:
        if gdat.sizepixl.size != gdat.numbside.size:
            print('')
            print('')
            print('')
            print('gdat.numbinst')
            print(gdat.numbinst)
            print('gdat.sizepixl')
            print(gdat.sizepixl)
            print('gdat.numbside')
            print(gdat.numbside)
            raise Exception('gdat.sizepixl.size != gdat.numbside.size')

    # size of the one side of the image in arcseconds
    gdat.sizeimag = gdat.sizepixl * gdat.numbside
    print('Size of the one side of the image [arcsec]')
    print(gdat.sizeimag)

    # global object for reference variables
    gdat.refr = tdpy.util.gdatstrt()
    # list of reference catalogs
    gdat.refr.lablcatl = []
    
    if not 'simutargsynt' in gdat.liststrgtypedata and gdat.nametarg != 'Earth':
        # temp -- check that the closest TIC to a given TIC is itself
        if gdat.maxmradiquer is None:
            # temp
            gdat.maxmradiquer = 0.1 * np.amax(gdat.sizepixl * np.sqrt(2.) * (gdat.numbside + 4) / 2.)
        print('gdat.maxmradiquer')
        print(gdat.maxmradiquer)
        print('gdat.sizepixl')
        print(gdat.sizepixl)
        print('Querying the TIC within %.3g arcseconds around the source with the MAST keyword %s and maximum Tmag %g...' \
                                                                                    % (gdat.maxmradiquer, gdat.strgmast, gdat.maxmtmagcatl))
        catalogData = astroquery.mast.Catalogs.query_criteria(coordinates=gdat.strgmast, radius='%ds' % gdat.maxmradiquer, catalog="TIC", Tmag=[-20., gdat.maxmtmagcatl])
        print('Found %d TIC sources within %.3g arcseconds.' % (len(catalogData), gdat.maxmradiquer))
        if len(catalogData) == 0:
            raise Exception('')
        gdat.refr.lablcatl += ['TIC']
    
    if gdat.nametarg == 'Earth':
        gdat.refr.lablcatl += ['CustDSCVR']

    if gdat.catlextr is not None:
        gdat.refr.lablcatl.extend(gdat.lablcatlextr)

    gdat.refr.numbcatl = len(gdat.refr.lablcatl)
    gdat.refr.indxcatl = np.arange(gdat.refr.numbcatl)
    
    print('gdat.refr.numbcatl')  
    print(gdat.refr.numbcatl)

    gdat.refr.liststrgfeatbase = [[] for q in gdat.refr.indxcatl]
    for q in gdat.refr.indxcatl:
        gdat.refr.liststrgfeatbase[q] += ['labl', 'cntsesti']
        if not 'simutargsynt' in gdat.liststrgtypedata:
            gdat.refr.liststrgfeatbase[q] += ['rasc', 'decl']
            if gdat.refr.lablcatl[q] == 'TIC':
                gdat.refr.liststrgfeatbase[q] += ['TICID', 'tmag', 'pmde', 'pmra', 'rascorig', 'declorig']
    gdat.refr.liststrgfeat = [[] for q in gdat.refr.indxcatl]
    for q in gdat.refr.indxcatl:
        gdat.refr.liststrgfeat[q] = gdat.refr.liststrgfeatbase[q] + ['xpos', 'ypos']
    
    gdat.refr.catlbase = [dict() for q in gdat.refr.indxcatl]
    gdat.refr.numbpntsbase = np.empty(gdat.refr.numbcatl, dtype=int)
    gdat.refr.indxpntsbase = [[] for q in np.arange(gdat.refr.numbcatl)]
    
    if gdat.strgtarg == 'Earth':
        gdat.refr.catlbase[0]['rasc'] = 0.
        gdat.refr.catlbase[0]['decl'] = 0.
    
    if gdat.cntsestitarg is None:
        gdat.cntsestitarg = 10.

    if gdat.tmagtarg is None:
        gdat.tmagtarg = 10.

    if gdat.pmdetarg is None:
        gdat.pmdetarg = 0.

    if gdat.pmratarg is None:
        gdat.pmratarg = 0.

    if gdat.rascorigtarg is None:
        gdat.rascorigtarg = gdat.rasctarg

    if gdat.declorigtarg is None:
        gdat.declorigtarg = gdat.decltarg
    
    if gdat.booldiag:
        if 'simutargsynt' in gdat.liststrgtypedata:
            for p in gdat.indxinst:
                if gdat.liststrgtypedata[p] != 'simutargsynt':
                    print('')
                    print('')
                    print('')
                    raise Exception('IF gdat.liststrgtypedata contains simutargsynt then all instruments should be simutargsynt.')

    gdat.boolsimuanyy = False
    for p in gdat.indxinst:
        if 'simu' in gdat.liststrgtypedata[p]:
            gdat.boolsimuanyy = True
    
    # Boolean flag indicating if the target is a synthetic simulated target
    gdat.boolsimutargsynt = 'simutargsynt' in gdat.liststrgtypedata
        
    if not gdat.boolsimutargsynt:
        for q in gdat.refr.indxcatl:
            if gdat.refr.lablcatl[q] == 'TIC':
                print('Constructing reference catalog %d...' % q)
                gdat.refr.numbpntsbase[q] = catalogData[:]['Tmag'].size
        
    for q in gdat.refr.indxcatl:
        
        if not gdat.boolsimutargsynt:
            gdat.refr.numbpntsbase[q] = catalogData[:]['Tmag'].size
            gdat.refr.numbpntsbase[q] = catalogData[:]['Tmag'].size
        
        if 'simutargpartinje' in gdat.liststrgtypedata:
            gdat.true.rasctarg = (gdat.maxmnumbside - 1.) / 2.
            gdat.true.decltarg = (gdat.maxmnumbside - 1.) / 2.
        
        if gdat.true.tmagneig.size > 0:
            # generate neighbors within 0.5 pixels of the edges
            gdat.true.velxneig = np.zeros(gdat.true.tmagneig.size)
            gdat.true.velyneig = np.zeros(gdat.true.tmagneig.size)
            gdat.refr.catlbase[q]['tmag'] = np.concatenate((np.array([gdat.true.tmagtarg]), gdat.true.tmagneig))
            if 'simutargsynt' in gdat.liststrgtypedata:
                gdat.refr.catlbase[q]['xpos'] = np.concatenate((np.array([gdat.true.xpostarg]), gdat.true.xposneig))
                gdat.refr.catlbase[q]['ypos'] = np.concatenate((np.array([gdat.true.ypostarg]), gdat.true.yposneig))
                gdat.refr.catlbase[q]['velx'] = np.concatenate((np.array([gdat.true.velxtarg]), gdat.true.velxneig))
                gdat.refr.catlbase[q]['vely'] = np.concatenate((np.array([gdat.true.velytarg]), gdat.true.velyneig))
            elif 'simutargpartsynt' in gdat.liststrgtypedata:
                gdat.refr.catlbase[q]['rasc'] = catalogData[:]['ra']
                gdat.refr.catlbase[q]['decl'] = catalogData[:]['dec']
                gdat.refr.catlbase[q]['velr'] = 0. * catalogData[:]['ra']
                gdat.refr.catlbase[q]['veld'] = 0. * catalogData[:]['dec']
            elif 'simutargpartinje' in gdat.liststrgtypedata:
                gdat.refr.catlbase[q]['rasc'] = np.concatenate((np.array([gdat.true.rasctarg]), catalogData[:]['ra']))
                gdat.refr.catlbase[q]['decl'] = np.concatenate((np.array([gdat.true.decltarg]), catalogData[:]['dec']))
                gdat.refr.catlbase[q]['velr'] = np.concatenate((np.array([gdat.true.velrtarg]), 0. * catalogData[:]['ra']))
                gdat.refr.catlbase[q]['veld'] = np.concatenate((np.array([gdat.true.veldtarg]), 0. * catalogData[:]['dec']))
            else:   
                raise Exception('')
        else:
            gdat.refr.catlbase[q]['rasc'] = np.array([gdat.true.rasctarg])
            gdat.refr.catlbase[q]['decl'] = np.array([gdat.true.decltarg])
            gdat.refr.catlbase[q]['tmag'] = np.array([gdat.true.tmagtarg])
            gdat.refr.catlbase[q]['velx'] = np.array([gdat.true.velxtarg])
            gdat.refr.catlbase[q]['vely'] = np.array([gdat.true.velytarg])
        
        # labels
        if gdat.boolsimutargsynt:
            gdat.refr.catlbase[q]['labl'] = np.empty(gdat.refr.catlbase[q]['xpos'].size, dtype=object)
            for k in np.arange(gdat.refr.catlbase[q]['labl'].size):
                gdat.refr.catlbase[q]['labl'][k] = '%d' % k
    
        if not gdat.boolsimutargsynt:
            
            offs = 0
            #if gdat.boolsimutargsynt or 'simutargpartinje':
            #    offs = 1
            
            for name in gdat.refr.liststrgfeatbase[q]:
                gdat.refr.catlbase[q][name] = np.empty(gdat.refr.numbpntsbase[q])
            gdat.refr.catlbase[q]['TICID'] = np.empty(gdat.refr.numbpntsbase[q], dtype=int)
            gdat.refr.catlbase[q]['labl'] = np.empty(gdat.refr.numbpntsbase[q], dtype=object)

            if gdat.refr.lablcatl[q] == 'TIC':
                gdat.refr.catlbase[q]['tmag'][offs:] = catalogData[:]['Tmag']
                gdat.refr.catlbase[q]['rasc'][offs:] = catalogData[:]['ra']
                gdat.refr.catlbase[q]['decl'][offs:] = catalogData[:]['dec']
                gdat.refr.catlbase[q]['TICID'][offs:] = catalogData[:]['ID']
                gdat.refr.catlbase[q]['pmde'][offs:] = catalogData[:]['pmDEC']
                gdat.refr.catlbase[q]['pmra'][offs:] = catalogData[:]['pmRA']
                gdat.refr.catlbase[q]['rascorig'][offs:] = catalogData[:]['RA_orig']
                gdat.refr.catlbase[q]['declorig'][offs:] = catalogData[:]['Dec_orig']
                for k in np.arange(catalogData[:]['ID'].size):
                    gdat.refr.catlbase[q]['labl'][offs+k] = 'TIC %s' % catalogData[:]['ID'][k]
        
        # estimate the counts
        for p in gdat.indxinst:
            if gdat.liststrginst[p].startswith('TESS') or gdat.liststrginst[p] == 'TGEO-IR':
                offszero = 20.4
            elif gdat.liststrginst[p] == 'TGEO-UV':
                offszero = 24.
            elif gdat.liststrginst[p] == 'TGEO-VIS':
                offszero = 24.
            elif gdat.liststrginst[p] == 'ULTRASAT':
                offszero = 22.4
            else:
                print('')
                print('')
                print('')
                print('p')
                print(p)
                print('gdat.liststrginst')
                print(gdat.liststrginst)
                raise Exception('gdat.liststrginst[p] undefined.')

            gdat.refr.catlbase[q]['cntsesti'] = 10**(-(gdat.refr.catlbase[q]['tmag'] - offszero) / 2.5)
            
        # add the target to the first reference catalog the target is a coordinate on the sky
        if gdat.typetarg == 'posi' and q == 0:
            for strgfeat in gdat.refr.liststrgfeatbase[q]:
                temp = np.copy(gdat.refr.catlbase[0][strgfeat])
                sizeprim = gdat.refr.catlbase[0][strgfeat].size + 1
                if strgfeat == 'labl':
                    gdat.refr.catlbase[0][strgfeat] = np.empty(sizeprim, dtype=object)
                else:
                    gdat.refr.catlbase[0][strgfeat] = np.empty(sizeprim)
                gdat.refr.catlbase[0][strgfeat][0] = getattr(gdat, strgfeat + 'targ')
                gdat.refr.catlbase[0][strgfeat][1:] = temp
        gdat.refr.numbpntsbase[q] = gdat.refr.catlbase[q]['labl'].size    
        gdat.refr.indxpntsbase[q] = np.arange(gdat.refr.numbpntsbase[q])
            
    if not gdat.boolsimutargsynt:
        
        if gdat.typetarg == 'TICID' or gdat.typetarg == 'TOIID' or gdat.typetarg == 'mast':
            # ensure that the first source is the target
            gdat.ticitarg = int(catalogData[0]['ID'])
            gdat.rasctarg = catalogData[0]['ra']
            gdat.decltarg = catalogData[0]['dec']
            gdat.tmagtarg = catalogData[0]['Tmag']
            #if catalogData[0]['dstArcSec'] > 0.1:
            #    print('The closest source returned by the MAST search is %g arcsec away from the target and may not be the correct one!' % catalogData[0]['dstArcSec'])
        if gdat.rasctarg is not None:
            objticrs = astropy.coordinates.SkyCoord(ra=gdat.rasctarg*astropy.units.degree, \
                                               dec=gdat.decltarg*astropy.units.degree, frame='icrs')
            gdat.lgalttarg = objticrs.galactic.l
            gdat.bgalttarg = objticrs.galactic.b
            gdat.laecttarg = objticrs.barycentricmeanecliptic.lon.degree
            gdat.beecttarg = objticrs.barycentricmeanecliptic.lat.degree
            
            print('gdat.ticitarg')
            print(gdat.ticitarg)
            print('gdat.rasctarg')
            print(gdat.rasctarg)
            print('gdat.decltarg')
            print(gdat.decltarg)
            if gdat.typetarg == 'TICID' or gdat.typetarg == 'TOIID' or gdat.typetarg == 'mast':
                print('gdat.tmagtarg')
                print(gdat.tmagtarg)
    
    if gdat.labltarg is None:
        if gdat.typetarg == 'mast':
            gdat.labltarg = gdat.strgmast
        elif gdat.typetarg == 'TOIID':
            gdat.labltarg = 'TOI %d' % gdat.toiitarg
        elif gdat.typetarg == 'TICID':
            gdat.labltarg = 'TIC %d' % gdat.ticitarg
        elif gdat.typetarg == 'posi':
            gdat.labltarg = 'RA=%.4g, DEC=%.4g' % (gdat.rasctarg, gdat.decltarg)
        elif gdat.typetarg == 'sols':
            gdat.labltarg = gdat.nametarg
        else:
            raise Exception('A label must be provided for the target when data are simulated based on a generative model.')
    if gdat.strgtarg is None:
        gdat.strgtarg = ''.join(gdat.labltarg.split(' '))
    
    print('List of instruments:')
    print(gdat.liststrginst)
    print('Target label: %s' % gdat.labltarg) 
    print('Output folder name: %s' % gdat.strgtarg) 
    if not gdat.boolsimutargsynt:
        if gdat.typetarg != 'sols':
            print('RA and DEC: %g %g' % (gdat.rasctarg, gdat.decltarg))
        if gdat.typetarg == 'TICID' or gdat.typetarg == 'mast':
            print('Tmag: %g' % gdat.tmagtarg)
   
    print('PSF inference type: %s' % gdat.typepsfninfe)
    print('PSF model shape type: %s' % gdat.fitt.typepsfnshap)
    print('PSF subpixel interpolation type: %s' % gdat.typepsfnsubp)
    
    if gdat.strgclus is None:
        gdat.strgclus = ''
    else:
        gdat.strgclus += '/'
    if gdat.strgruns is None:
        gdat.strgruns = ''
    else:
        gdat.strgruns = '/' + gdat.strgruns
    if gdat.pathtarg is None:
        gdat.pathtarg = gdat.pathbase + '%s%s%s/' % (gdat.strgclus, gdat.strgtarg, gdat.strgruns)
    
    gdat.pathdatatarg = gdat.pathtarg + 'data/'
    gdat.pathvisutarg = gdat.pathtarg + 'visuals/'
    gdat.pathclus = gdat.pathbase + '%s' % gdat.strgclus
    gdat.pathdataclus = gdat.pathclus + 'data/'
    gdat.pathvisuclus = gdat.pathclus + 'visuals/'
    
    if gdat.boolplot:
        os.system('mkdir -p %s' % gdat.pathvisulygo)
        os.system('mkdir -p %s' % gdat.pathvisutarg)
        os.system('mkdir -p %s' % gdat.pathvisuclus)
    os.system('mkdir -p %s' % gdat.pathdatalygo)
    os.system('mkdir -p %s' % gdat.pathdatatarg)
    os.system('mkdir -p %s' % gdat.pathdataclus)
    
    # create a separate folder to place the PSF fit output
    gdat.pathvisutargsexp = gdat.pathvisutarg + 'sexp/'
    os.system('mkdir -p %s' % gdat.pathvisutargsexp)
   
    # header that will be added to the output CSV files
    gdat.strgheadtarg = 'time [BJD], relative flux, relative flux error'
    
    for p in gdat.indxinst:
        print('Number of pixels on a side: %d' % gdat.numbside[p])

    if gdat.strgtarg is None:
        gdat.strgtarg = '%016d' % gdat.ticitarg

    # construct the string describing the data configuration
    gdat.strgcnfg = '%s' % (gdat.strgtarg)
    
    gdat.dictindxinst = dict()
    for name in ['TESS', 'TGEO-IR', 'TGEO-VIS', 'TGEO-UV', 'ULTRASAT', 'LSST']:
        indx = np.where(gdat.liststrginst == name)[0]
        if len(indx) > 0:
            gdat.dictindxinst[name] = indx[0]
    
    # list of IDs for pointings (for TESS, this is the sector ID)
    gdat.listipnt = [[] for p in gdat.indxinst]
    gdat.listtcam = []
    gdat.listtccd = []
    
    gdat.dictoutp = dict()
    gdat.dictoutp['arryrflx'] = dict()
    for p in gdat.indxinst:
        if gdat.liststrgtypedata[p] == 'simutargpartsynt' or gdat.liststrgtypedata[p] == 'simutargsynt':
            gdat.listipnt[p] = np.array([1])
            gdat.listtcam = np.array([0])
            gdat.listtccd = np.array([0])
    
        if 'TGEO-IR' in gdat.liststrginst or 'TGEO-VIS' in gdat.liststrginst:
            gdat.listipnt[p] = np.array([1])
            #gdat.listtcam = [1]
            #gdat.listtccd = [1]
    
    gdat.listtcamffim = [[] for p in gdat.indxinst]
    gdat.listtccdffim = [[] for p in gdat.indxinst]
    
    booltessanyy = False
    for p in gdat.indxinst:
        if gdat.liststrginst[p].startswith('TESS') and gdat.liststrginst[p] != 'TESSCam':
            booltessanyy = True
    
    gdat.boolterm = False

    if booltessanyy:
        # get TESS FFI data via TESSCut, either using the stored HDU lists for each sector or online
        gdat.pathdatatargtcut = gdat.pathdatatarg + 'TESSCut/'
        if os.path.exists(gdat.pathdatatargtcut):
            print('Looking for TESSCut FITS files in %s...' % gdat.pathdatatargtcut)
            listname = fnmatch.filter(os.listdir(gdat.pathdatatargtcut), 'tess-*_astrocut.fits')
            if len(listname) == 0:
                print('Did not find any TESSCut FITS files.')
        else:
            print('The path in which TESSCut FITS files are expected does not exist: %s' % gdat.pathdatatargtcut)
        
        if not os.path.exists(gdat.pathdatatargtcut) or len(listname) == 0:
            
            os.system('mkdir -p %s' % gdat.pathdatatargtcut)
            
            timeinit = timemodu.time()

            #down_tcut(gdat)

            import multiprocessing

            # time out
            timetout = 1200 # [sec]

            if __name__ == '__main__':
                objtproc = multiprocessing.Process(target=down_tcut, args=(gdat, ))
                
                objtproc.start()
                
                # Wait until process finishes
                objtproc.join(timetout)

                # check if the process is still active
                if objtproc.is_alive():
                    print('\nThe download call to TESSCut is still running after %d seconds, which may indicate a problem. Will time out and kill it...' % timetout)

                    # Terminate - may not work if process is stuck for good
                    #objtproc.terminate()
                    objtproc.kill()

                    objtproc.join()
                    
                    gdat.boolterm = True

                else:
                    timefinl = timemodu.time()
                    print('Successfully called TESSCut in %g seconds.' % (timefinl - timeinit))
                            
            strgkeyy = "tess-s*_%.6f_%.6f_%dx%d_astrocut.fits" % (gdat.rasctarg, gdat.decltarg, gdat.numbsidedefa, gdat.numbsidedefa)
            liststrgfile = fnmatch.filter(os.listdir(gdat.pathdatatargtcut), strgkeyy)
            
            listhdundatatemp = []
            for strgfile in liststrgfile:
                path = gdat.pathdatatargtcut + strgfile
                print('Reading from %s...' % path)
                listhdundatatemp.append(astropy.io.fits.open(path))
            
            #for oo in range(len(listhdundatatemp)):
                #path = gdat.pathdatatargtcut + 'ts%02d.fits' % listhdundatatemp[oo][0].header['SECTOR']
                #astropy.io.fits.HDUList(listhdundatatemp[oo]).writeto(path)
            
        else:
            listhdundatatemp = []
            for name in listname:
                path = gdat.pathdatatargtcut + name
                print('Reading from %s...' % path)
                listhdundatatemp.append(astropy.io.fits.open(path))
        
        gdat.listhdundataffimtess = []
        gdat.listtsecffim = []
        gdat.listtcamffim = []
        gdat.listtccdffim = []
        for o, hdundata in enumerate(listhdundatatemp):
            gdat.listtsecffim.append(hdundata[0].header['SECTOR'])
            gdat.listtcamffim.append(hdundata[0].header['CAMERA'])
            gdat.listtccdffim.append(hdundata[0].header['CCD'])
            gdat.listhdundataffimtess.append(hdundata)
        
        # sort the poiting ID, Camera, CCD, and HDU lists according to pointing ID
        ## get an array holding the indices that would sort
        indxsort = np.argsort(np.array(gdat.listtsecffim))
        ## copy the lists to temporary variables
        gdat.listtsecffimtemp = list(gdat.listtsecffim)
        gdat.listtcamffimtemp = list(gdat.listtcamffim)
        gdat.listtccdffimtemp = list(gdat.listtccdffim)
        gdat.listhdundataffimtemp = list(gdat.listhdundataffimtess)
        ## write onto the lists with the sorted order 
        for kk, indxtemp in enumerate(indxsort):
            gdat.listtsecffim[kk] = gdat.listtsecffimtemp[indxtemp]
            gdat.listtcamffim[kk] = gdat.listtcamffimtemp[indxtemp]
            gdat.listtccdffim[kk] = gdat.listtccdffimtemp[indxtemp]
            gdat.listhdundataffimtess[kk] = gdat.listhdundataffimtemp[indxtemp]

        print('gdat.listtsecffim')
        print(gdat.listtsecffim)
        print('gdat.listtcamffim')
        print(gdat.listtcamffim)
        print('gdat.listtccdffim')
        print(gdat.listtccdffim)
        
        gdat.listtsecspoc = []
        gdat.listtcamspoc = []
        gdat.listtccdspoc = []
        print('boolutiltpxf')
        print(boolutiltpxf)
        if boolutiltpxf:
            
            # get the list of sectors for which TPF data are available
            print('Retrieving the list of available TESS sectors for which there is higher-cadence SPOC TPF data...')
            # get observation tables
            listtablobsv = nicomedia.retr_listtablobsv(gdat.strgmast)
            numbtablobsv = len(listtablobsv)
            print('%d observation tables have been found.' % numbtablobsv)
            listprodspoc = []
            for k, tablobsv in enumerate(listtablobsv):
                
                listprodspoctemp = astroquery.mast.Observations.get_product_list(tablobsv)
                
                if listtablobsv['distance'][k] > 0:
                    continue

                strgdesc = ['Target pixel files', 'Light curves']
                listprodspoctemp = astroquery.mast.Observations.filter_products(listprodspoctemp, description=strgdesc)
                for a in range(len(listprodspoctemp)):
                    boolfasttemp = listprodspoctemp[a]['obs_id'].endswith('fast')
                    boolgood = boolfasttemp == gdat.boolfasttpxf

                    if boolgood and listprodspoctemp[a]['description'] == 'Target pixel files':
                        tsec = int(listprodspoctemp[a]['obs_id'].split('-')[1][1:])
                        gdat.listtsecspoc.append(tsec) 
                        
                        print('temp: assigning dummy Cam and CCD to the target for this sector.')
                        gdat.listtcamspoc.append(-1) 
                        gdat.listtccdspoc.append(-1) 
                        
                        listprodspoc.append(listprodspoctemp)
            
            gdat.listtsecspoc = np.array(gdat.listtsecspoc, dtype=int)
            gdat.listtcamspoc = np.array(gdat.listtcamspoc, dtype=int)
            gdat.listtccdspoc = np.array(gdat.listtccdspoc, dtype=int)
            
            print('gdat.listtcamspoc')
            print(gdat.listtcamspoc)

            indx = np.argsort(gdat.listtsecspoc)
            gdat.listtsecspoc = gdat.listtsecspoc[indx]
            gdat.listtcamspoc = gdat.listtcamspoc[indx]
            gdat.listtccdspoc = gdat.listtccdspoc[indx]
            
            print('gdat.listtcamspoc')
            print(gdat.listtcamspoc)

            gdat.numbpoinspoc = gdat.listtsecspoc.size

        if len(gdat.listtsecspoc) > 0:
            gdat.indxtsecspoc = np.arange(gdat.numbpoinspoc)
            
            # download data from MAST
            os.system('mkdir -p %s' % gdat.pathdatatarg)
        
            print('Downloading SPOC data products...')
            
            listpathdownspoc = []

            listpathdownspoclcur = []
            listpathdownspoctpxf = []
            for k in range(len(listprodspoc)):
                manifest = astroquery.mast.Observations.download_products(listprodspoc[k], download_dir=gdat.pathdatatarg)
                listpathdownspoclcur.append(manifest['Local Path'][0])
                listpathdownspoctpxf.append(manifest['Local Path'][1])

            ## make sure the list of paths to sector files are time-sorted
            listpathdownspoc.sort()
            listpathdownspoclcur.sort()
            listpathdownspoctpxf.sort()
            
            ## read SPOC TPFs
            gdat.listhdundataspoc = [[] for o in gdat.indxtsecspoc]
            gdat.indxtimegoodspoc = [[] for o in gdat.indxtsecspoc]
            for oo in gdat.indxtsecspoc:
                print('oo')
                print(oo)
                print('listpathdownspoctpxf')
                print(listpathdownspoctpxf)
                print('gdat.listtcamspoc')
                print(gdat.listtcamspoc)
                gdat.listhdundataspoc[oo], gdat.indxtimegoodspoc[oo], gdat.listtsecspoc[oo], gdat.listtcamspoc[oo], \
                                                                gdat.listtccdspoc[oo] = miletos.read_tesskplr_file(listpathdownspoctpxf[oo])
                
                print('gdat.listhdundataspoc[oo]')
                print(gdat.listhdundataspoc[oo])

                if not np.isfinite(gdat.listhdundataspoc[oo][1].data['TIME'][gdat.indxtimegoodspoc[oo]]).all():
                    raise Exception('')
            
            print('gdat.listhdundataspoc')
            print(gdat.listhdundataspoc)
            print('gdat.listtsecspoc')
            print(gdat.listtsecspoc)
            print('gdat.listtcamspoc')
            print(gdat.listtcamspoc)
            print('gdat.listtccdspoc')
            print(gdat.listtccdspoc)

        # merge SPOC TPF and FFI sector lists
        gdat.listtsecffim = np.array(gdat.listtsecffim, dtype=int)
        
        if len(gdat.listtsecspoc) == 0:
            gdat.listtsecconc = gdat.listtsecffim
        else:
            gdat.listtsecconc = np.unique(np.concatenate((gdat.listtsecffim, gdat.listtsecspoc), dtype=int))
        
        if gdat.listtsecsele is not None:
            
            if isinstance(gdat.listtsecsele, list):
                gdat.listtsecsele = np.array(gdat.listtsecsele)

            if gdat.booldiag:
                if np.setdiff1d(gdat.listtsecsele, gdat.listtsecconc).size > 0:
                    print('')
                    print('')
                    print('')
                    print('gdat.listtsecsele')
                    print(gdat.listtsecsele)
                    print('gdat.listtsecconc')
                    print(gdat.listtsecconc)
                    #raise Exception
                    print('Warning!! gdat.listtsecsele has a sector not available.')

            print('Taking only selected sectors...')
            print('gdat.listtsecsele')
            print(gdat.listtsecsele)
            gdat.listtsecconc = [tsec for tsec in gdat.listtsecconc if tsec in gdat.listtsecsele]
            
        print('gdat.listtsecconc')
        print(gdat.listtsecconc)
        
        gdat.listtsec = []
        gdat.listtccd = []
        gdat.listtcam = []
        
        for k in range(len(gdat.listtsecconc)):
            indx = np.where(gdat.listtsecspoc == gdat.listtsecconc[k])[0]
            print('gdat.listtsecconc[k]')
            print(gdat.listtsecconc[k])
            if indx.size > 0:
                gdat.listtsec.append(gdat.listtsecspoc[indx[0]])
                gdat.listtcam.append(gdat.listtcamspoc[indx[0]])
                gdat.listtccd.append(gdat.listtccdspoc[indx[0]])
            else:
                indx = np.where(gdat.listtsecffim == gdat.listtsecconc[k])[0]
                gdat.listtsec.append(gdat.listtsecffim[indx[0]])
                gdat.listtcam.append(gdat.listtcamffim[indx[0]])
                gdat.listtccd.append(gdat.listtccdffim[indx[0]])
                
                print('gdat.listtcamffim')
                print(gdat.listtcamffim)
                print('gdat.listtsecconc[k]')
                print(gdat.listtsecconc[k])
                print('indx')
                print(indx)
                print('gdat.listtsecffim')
                print(gdat.listtsecffim)
                print('gdat.listtcam')
                print(gdat.listtcam)
            print('')

        if gdat.booldiag:
            if len(gdat.listtsec) == 0:
                print('')
                print('')
                print('')
                print('gdat.listtsec')
                print(gdat.listtsec)
                raise Exception('')

        gdat.listtsec = np.array(gdat.listtsec)
        gdat.listtcam = np.array(gdat.listtcam)
        gdat.listtccd = np.array(gdat.listtccd)
                
    for p in gdat.indxinst:
        
        if gdat.liststrginst[p] == 'TESS':
            gdat.listipnt[p] = gdat.listtsec
            
            if gdat.booldiag:
                if len(gdat.listtsec) == 0:
                    print('')
                    print('')
                    print('')
                    print('gdat.listtsec')
                    print(gdat.listtsec)
                    raise Exception('')

        if isinstance(gdat.listipnt[p], list):
            gdat.listipnt[p] = np.array(gdat.listipnt[p], dtype=int)

    # number of pointings
    gdat.numbpoin = np.ones(gdat.numbinst, dtype=int)
    for p in gdat.indxinst:
        gdat.numbpoin[p] = gdat.listipnt[p].size
    
    # TESS-specific
    if booltessanyy:
        
        gdat.dictoutp['listtsec'] = gdat.listtsec
        gdat.dictoutp['listtcam'] = gdat.listtcam
        gdat.dictoutp['listtccd'] = gdat.listtccd
        
        ## current pointing ID of TESS
        # temp: automatize this
        gdat.tseccurr = 65

        if gdat.booldiag:
            if len(gdat.listtsec) != len(gdat.listtcam):
                print('')
                print('')
                print('')
                print('gdat.listipnt[p]')
                print(gdat.listipnt[p])
                print('gdat.listtcam')
                print(gdat.listtcam)
                raise Exception('len(gdat.listipnt[p]) != len(gdat.listtcam)')

        # Boolean flag to indicate TESS data in the "past"
        ## if False, it means the data must be simulated
        gdat.booltesspast = gdat.listtsec < gdat.tseccurr
            
    if gdat.boolterm:
        return gdat.dictoutp

    if (gdat.numbpoin == 0).any():
        print('')
        print('')
        print('')
        for p in gdat.indxinst:
            print('p')
            print(p)
            print('gdat.liststrginst[p]')
            print(gdat.liststrginst[p])
            print('gdat.listipnt[p]')
            print(gdat.listipnt[p])
            print('')
        print('gdat.numbpoin')
        print(gdat.numbpoin)
        raise Exception('(gdat.numbpoin == 0).any()')
        
    gdat.indxtsec = [[] for p in gdat.indxinst]
    for p in gdat.indxinst:
        print('gdat.numbpoin[p]')
        print(gdat.numbpoin[p])
        gdat.indxtsec[p] = np.arange(gdat.numbpoin[p])
    
    if gdat.listtime is None:
        gdat.listtime = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    
    if gdat.listnameanls is None:
        gdat.listnameanls = ['psfn']
        if gdat.boolphotaper:
            gdat.listnameanls.append('aper')
            gdat.indxanlsaper = 1
    
    gdat.indxanlsaper = 0

    gdat.numbanls = len(gdat.listnameanls)
    gdat.indxanls = np.arange(gdat.numbanls)
    for e in gdat.indxanls:
        gdat.dictoutp['arryrflx'][gdat.listnameanls[e]] = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]

    gdat.offsposi = 0.25
    gdat.listoffsposi = np.linspace(-gdat.offsposi, gdat.offsposi, 3)
    gdat.numboffs = gdat.listoffsposi.size
    gdat.indxoffs = np.arange(gdat.numboffs)
    
    gdat.gainphot = np.empty(gdat.numbinst)

    gdat.fitt.tserfile = [[[[[[] for y in gdat.indxoffs] for x in gdat.indxoffs] for e in gdat.indxanls] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.refr.numbpnts = [[[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in gdat.refr.indxcatl]
    gdat.typecade = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.booltpxf = [[] for p in gdat.indxinst]
    for p in gdat.indxinst:
        if gdat.booldiag:
            if isinstance(gdat.listipnt, int) or gdat.listipnt[p].size == 0:
                raise Exception('')

        gdat.booltpxf[p] = np.zeros(gdat.numbpoin[p], dtype=bool)
        
        if not gdat.boolsimutargsynt:
        
            if gdat.boolinptnumbside and gdat.numbside[p] != 11:
                print('Will not be using TPFs since number of pixels along a side is not 11.')
            else:
                # determine whether sectors have TPFs
                gdat.booltpxf[p] = ~tdpy.retr_boolsubb(gdat.listipnt[p], gdat.listtsecspoc)

            if gdat.booldiag:
                for o in gdat.indxtsec[p]:
                    if not gdat.booltesspast[o] and gdat.liststrgtypedata[p] == 'obsd':
                        print('')
                        print('')
                        print('')
                        print('gdat.listipnt[p]')
                        print(gdat.listipnt[p])
                        print('gdat.tseccurr')
                        print(gdat.tseccurr)
                        print('gdat.liststrginst[p]')
                        print(gdat.liststrginst[p])
                        print('gdat.liststrgtypedata[p]')
                        print(gdat.liststrgtypedata[p])
                        raise Exception('gdat.booltesspast[o] is False, so the data must be simulated.')

                    if gdat.booltpxf[p][o]:
                        if gdat.boolinptnumbside:
                            raise Exception('')
        
        print('gdat.indxtsec[p]')
        print(gdat.indxtsec[p])
        print('gdat.booltpxf[p]')
        print(gdat.booltpxf[p])
        
        # determine the cadence
        if gdat.liststrginst[p] == 'TESS' or gdat.liststrginst[p].startswith('TGEO') or gdat.liststrginst[p] == 'ULTRASAT' or gdat.liststrginst[p] == 'TESSCam':
            
            # set up the gain
            gdat.gainphot[p] = 1.

            for o, tsec in enumerate(gdat.listipnt[p]):
                if gdat.liststrginst[p] == 'TESS':
                    if gdat.booltpxf[p][o]:
                        # temp does not work with 20sc
                        gdat.typecade[p][o] = '2min'
                    else:
                        if tsec >= 56:
                            gdat.typecade[p][o] = '200s'
                        elif tsec >= 27:
                            gdat.typecade[p][o] = '10mn'
                        else:
                            gdat.typecade[p][o] = '30mn'
                else:
                    gdat.typecade[p][o] = '200s'
        elif gdat.liststrginst[p].startswith('LSST'):
            pass
        else:
            print('gdat.liststrginst[p]')
            print(gdat.liststrginst[p])
            raise Exception('')

        if not gdat.boolsimutargsynt:
            if gdat.booltesspast[o]:
                if gdat.numbpoin[p] == 0:
                    print('No data have been retrieved for instrument %s.' % gdat.liststrginst[p])
                else:
                    print('%d sectors (unique pointings directions) of data retrieved for instrument %s.' % (gdat.numbpoin[p], gdat.liststrginst[p]))
    
                ## check for an earlier lygos run
                #for o in gdat.indxtsec:
                #    strgtsec = 'sc%02d' % gdat.listipnt[p][o]
                #    strgchun = retr_strgchun(gdat, p, o)
                #    for x in gdat.indxoffs:
                #        for y in gdat.indxoffs:
                #            updt_strgsave(gdat, strgchun, x, y, p, o)
                #            for e, nameanls in enumerate(gdat.listnameanls):
                #                pathsaverflxtarg = gdat.pathdatatarg + 'tser%s_%s%s.csv' % (gdat.strgnorm, nameanls, gdat.strgsave)
                #                gdat.dictoutp['pathsaverflxtarg_%s_%s' % (nameanls, strgtsec)] = pathsaverflxtarg
                #            
                #                if os.path.exists(pathsaverflxtarg):
                #                    print('Analysis of Sector %d previously completed...' % gdat.listipnt[p][o])
                #                    print('Reading from %s...' % pathsaverflxtarg)
                #                    gdat.fitt.tserfile[p][o][e][x][y] = np.loadtxt(pathsaverflxtarg, delimiter=',', skiprows=1)
        
    print('gdat.typecade')
    print(gdat.typecade)

    # read TESS PSF
    pathpsfn = gdat.pathdatalygo + 'tesspsfn/listpsfn.h5'
    print('Reading from %s...' % pathpsfn)
    objth5py = h5py.File(pathpsfn, 'r')
    listpsfn = objth5py.get('listpsfn')
    cntppsfntess = np.array(listpsfn)[0, 0, 0, 0, :, :]
    cntppsfntess /= np.sum(cntppsfntess)
    objth5py.close()
        
    #pathpsfn = gdat.pathdatalygo + 'tesspsfn.txt'
    #if not os.path.exists(pathpsfn):
    #    cmnd = 'tsig-psf --resolution 11 --id 4564 --show-contents > %s' % pathpsfn
    #    os.system(cmnd)
    ## read
    #cntppsfnusam = np.loadtxt(pathpsfn, delimiter=',')
    #cntppsfnusam = ndimage.convolve(cntppsfnusam, np.ones((11, 11)) / 121., mode='constant', cval=0.) * 121.
    #cntppsfnusam = cntppsfnusam[None, :, :]
    
    # size of the markers
    gdat.sizemrkrsour = 1.
    
    gdat.sizefigrside = 3.5
    
    gdat.sizefigrside /= gdat.factsizetextfigr
    
    gdat.sizefigrsing = (gdat.sizefigrside, gdat.sizefigrside)
    gdat.sizefigrdoubimag = (2 * gdat.sizefigrside, 2 * gdat.sizefigrside)
    gdat.sizefigrdoublcur = (2 * gdat.sizefigrside, gdat.sizefigrside)
    
    gdat.listtypecatlplot = ['fitt', 'refr', 'refrfitt']
    # setup regression
    if gdat.typepsfnsubp == 'regrcubipixl':
        gdat.numbregixaxi = gdat.numbside
        gdat.numbregiyaxi = gdat.numbside
        gdat.numbsidepsfn = 17

        gdat.numbsidepsfnusam = 13 * 9
        gdat.factusam = 9
        gdat.cntppsfnusam = cntppsfntess
        gdat.cntppsfnusam = ndimage.convolve(gdat.cntppsfnusam, np.ones((9, 9)), mode='constant', cval=0.)
        gdat.cntppsfnusam /= np.sum(gdat.cntppsfnusam)
        gdat.cntppsfnusam *= 81
        print('gdat.cntppsfnusam')
        summgene(gdat.cntppsfnusam)
        gdat.boolplotsave = False
   
        # make design matrix for each original pixel
        print('gdat.factusam')
        print(gdat.factusam)
        print('gdat.numbsidepsfnusam')
        print(gdat.numbsidepsfnusam)
        gdat.numbsidepsfn = int(gdat.numbsidepsfnusam / gdat.factusam)
        print('gdat.numbsidepsfn')
        print(gdat.numbsidepsfn)
        assert(gdat.numbsidepsfn % 2 == 1)
        
        # construct the design matrix for the pixel

        maxm = 1. / gdat.factusam
        arry = np.linspace(-maxm, maxm, gdat.factusam)
        x, y = np.meshgrid(arry, arry)
        x = x.flatten()
        y = y.flatten()
        A = np.column_stack([np.ones(gdat.factusam**2), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y])
        
        # number of regressors
        gdat.numbpararegr = A.shape[1]

        # regression coefficients for each pixel
        gdat.coefspix = np.zeros((gdat.numbpararegr, gdat.numbsidepsfn, gdat.numbsidepsfn))
        
        # loop over original psf pixels and regress the coefficients
        for a in np.arange(gdat.numbsidepsfn):
            for j in np.arange(gdat.numbsidepsfn):
                # solve p = matrdesi x coefspix for coefspix
                p = gdat.cntppsfnusam[a*gdat.factusam:(a+1)*gdat.factusam, j*gdat.factusam:(j+1)*gdat.factusam].flatten()
                gdat.coefspix[:, a, j] = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, p)) 
        
        if gdat.booldiag:
            if np.amax(gdat.coefspix) == 0.:
                raise Exception('')

        gdat.coefspix = gdat.coefspix[:, 1:12, 1:12]
        
        gdat.coefspix = gdat.coefspix.reshape(gdat.numbpararegr, gdat.numbside**2)
        
        if gdat.boolplotsave:
            figr, axis = plt.subplots()
            axis.imshow(gdat.cntppsfnusam, interpolation='nearest')
            plt.tight_layout()
            plt.savefig(gdat.pathdatartag + '%s_psfnusamp.%s' % (gdat.rtag, gdat.strgplotfile))
            plt.close()

    # exposure time
    if gdat.timeexpo is None:
        gdat.timeexpo = [[None for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    #for p in gdat.indxinst:
    #    for o in gdat.indxtsec[p]:
    #        if gdat.liststrginst[p] == 'TESS':
    #            if gdat.typecade[p][o] == '20sc':
    #                gdat.timeexpo[p][o] = 20. # [sec]
    #            elif gdat.typecade[p][o] == '2min':
    #                gdat.timeexpo[p][o] = 120. # [sec]
    #            elif gdat.typecade[p][o] == '10mn':
    #                gdat.timeexpo[p][o] = 1600. # [sec]
    #            elif gdat.typecade[p][o] == '30mn':
    #                gdat.timeexpo[p][o] = 1800. # [sec]
    #            else:
    #                print('gdat.typecade')
    #                print(gdat.typecade)
    #                raise Exception('')
    
    # list of labels for each pointing
    gdat.listlablpoin = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    for p in gdat.indxinst:
        for o in gdat.indxtsec[p]:
            gdat.listlablpoin[p][o] = 'Sector %d' % gdat.listipnt[p][o]
    
    gdat.refr.catl = [[[dict() for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in gdat.refr.indxcatl]
    gdat.refr.cequ = [[[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in gdat.refr.indxcatl]
    for q in gdat.refr.indxcatl:
        
        print('Number of sources in the reference catalog %d: %d' % (q, len(gdat.refr.catlbase[q]['tmag'])))
        for p in gdat.indxinst:
            
            if len(gdat.refr.catl[q][p]) != gdat.numbpoin[p]:
                print('gdat.numbpoin[p]')
                print(gdat.numbpoin[p])
                print('len(gdat.refr.catl[q][p)')
                print(len(gdat.refr.catl[q][p]))
                print('gdat.indxtsec')
                summgene(gdat.indxtsec)
                raise Exception('')
        
            for o in gdat.indxtsec[p]:
                for strgfeat in gdat.refr.liststrgfeatbase[q]:
                    gdat.refr.catl[q][p][o][strgfeat] = np.array(gdat.refr.catlbase[q][strgfeat])
    
    print('gdat.refr.catl')
    print(gdat.refr.catl)
    
    gdat.indxpntsbrgt = [[[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in gdat.refr.indxcatl]
    for q in gdat.refr.indxcatl:
        
        #gdat.refr.raticontsing = retr_raticontsing(gdat, gdat.refr.catlbase[q]['xpos'], gdat.refr.catlbase[q]['ypos'], \
        #                                                                            gdat.refr.catlbase[q]['cnts'], gdat.fitt.parapsfn)
        
        #if gdat.typetarg == 'TICID' or gdat.typetarg == 'TOIID' or gdat.typetarg == 'mast':
            
        #dmag = (gdat.refr.catlbase[q]['tmag'] - gdat.refr.catlbase[q]['tmag'][0]) / 
        if gdat.boolsimutargsynt:
            distsqrd = (gdat.refr.catlbase[q]['xpos'] - gdat.refr.catlbase[q]['xpos'][0])**2  + (gdat.refr.catlbase[q]['ypos'] - gdat.refr.catlbase[q]['ypos'][0])**2
        else:
            distsqrd = (gdat.refr.catlbase[q]['rasc'] - gdat.refr.catlbase[q]['rasc'][0])**2  + (gdat.refr.catlbase[q]['decl'] - gdat.refr.catlbase[q]['decl'][0])**2
        print('gdat.refr.catlbase[q]')
        print(gdat.refr.catlbase[q])
        print('distsqrd')
        print(distsqrd)
        fracblen = gdat.refr.catlbase[q]['cntsesti'] / gdat.refr.catlbase[q]['cntsesti'][0] * np.exp(-0.5*distsqrd/(1./3600.)**2)
        print('fracblen')
        print(fracblen)
        print('gdat.booldiag')
        print(gdat.booldiag)
        if gdat.booldiag:
            if fracblen[0] != 1.:
                raise Exception('')

        gdat.indxpntsbrgt[q] = np.where(fracblen > 1e-4)[0]
        
        gdat.numbrefrbrgt = gdat.indxpntsbrgt[q].size
        
        #magtcutt = gdat.refr.catlbase[q]['tmag'][0] + gdat.maxmdmag
        
        print('Number of reference sources that are bright & blended is %d.' % gdat.numbrefrbrgt)
     
    if gdat.booldiag:
        if len(gdat.indxpntsbrgt[0]) == 0:
            print('')
            print('')
            print('')
            print('gdat.indxpntsbrgt[0]')
            print(gdat.indxpntsbrgt[0])
            raise Exception('len(gdat.indxpntsbrgt[0]) == 0')
            
    #for p in gdat.indxinst:
    #print('Removing nearby sources that are too close separately for each instrument...')
    ## calculate angular distances
    #distangl = 180. * np.sqrt((gdat.refr.catl[q][p][o]['rasc'][None, :] - gdat.refr.catl[q][p][o]['rasc'][:, None])**2 + \
    #                   (gdat.refr.catl[q][p][o]['decl'][None, :] - gdat.refr.catl[q][p][o]['decl'][:, None])**2)
    #print('distangl')
    #print(distangl)
    #summgene(distangl)
    #indxnzer = np.where(distangl != 0)
    #while np.amin(distangl[indxnzer]) < 0.5:
    #    # find the two sources that are too close
    #    indxsort = np.argsort(distangl[indxnzer])
    #    indkill = indxsort[0]
    #    print('indxkill')
    #    print(indxkill)
    #    # determine the fainter one
    #    if gdat.refr.catl[q][p][o][strgfeat]['tmag'][indxsort[0]] < gdat.refr.catl[q][p][o][strgfeat]['tmag'][indx[1]]:
    #        indxkill = indxsort[1]
    #    
    #    # determine the new indcices (i.e., without the one to be killed)
    #    indxtotl = np.arange(gdat.refr.catl[q][p][o]['rasc'].size)
    #    indxneww = np.setdiff1d(indxtotl, np.array(indxkill))
    #    
    #    # remove the faint source
    #    for strgfeat in gdat.refr.liststrgfeat[q]:
    #        gdat.refr.catl[q][p][o][strgfeat] = gdat.refr.catl[q][p][o][strgfeat][indxneww]
    #    
    #    # recalculate the distances
    #    distangl = np.sqrt((gdat.refr.catl[q][p][o]['rasc'][None, :] - gdat.refr.catl[q][p][o]['rasc'][:, None])**2 + \
    #                       (gdat.refr.catl[q][p][o]['decl'][None, :] - gdat.refr.catl[q][p][o]['decl'][:, None])**2)
    
    gdat.refr.labltser = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.refr.numbtser = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.refr.indxtser = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    
    if gdat.refrlistlabltser is not None:
        gdat.refr.labltser = gdat.refrlistlabltser
    else:
        # determine what reference light curve is available for the sector
        for p in gdat.indxinst:
            for o in gdat.indxtsec[p]:
                if gdat.liststrgtypedata[p] != 'simutargsynt' and gdat.booltpxf[p][o]:
                    gdat.refr.labltser[p][o] += ['SPOC']
    
                # number of reference light curves
                gdat.refr.numbtser[p][o] = len(gdat.refr.labltser[p][o])
                gdat.refr.indxtser[p][o] = np.arange(gdat.refr.numbtser[p][o])
    
    gdat.refr.colrcatl = np.array(['r', 'orange', 'deepskyblue'], dtype=object)
    gdat.refr.colrcatl = gdat.refr.colrcatl[:gdat.refr.numbcatl]
    
    gdat.refr.colrlcur = np.array(['r', 'orange', 'b', 'g', 'olive'], dtype=object)
    
    # reference light curves
    gdat.refr.time = [[[[] for q in gdat.refr.indxtser[p][o]] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.refr.rflx = [[[[] for q in gdat.refr.indxtser[p][o]] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.refr.stdvrflx = [[[[] for q in gdat.refr.indxtser[p][o]] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    
    # temporal types of image plots
    ## medians
    listtypeplotcntp = []
    if gdat.boolplotcntp:
        listtypeplotcntp += ['tmed']
    # cadence frames
    if gdat.boolanim:
        listtypeplotcntp += ['anim']
        
    # write metadata to file
    #gdat.pathsavemetaglob = gdat.pathdatatarg + 'metatarg.csv'
    #dictmeta = dict()
    #print('Writing to %s...' % gdat.pathsavemetaglob)
    #objtfile = open(gdat.pathsavemetaglob, 'w')
    #for key, value in dictmeta.items():
    #    objtfile.write('%s,%g\n' % (key, value))
    #objtfile.close()

    # Boolean flag to indicate whether there is a reference time-series
    gdat.boolrefrtser = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    if gdat.refrarrytser is None:
        if gdat.boolsimutargsynt:
            for p in gdat.indxinst:
                for o in gdat.indxtsec[p]:
                    gdat.boolrefrtser[p][o] = False
        
        cntr = 0
        for p in gdat.indxinst:
            if gdat.liststrgtypedata[p] != 'simutargsynt':
                for o in gdat.indxtsec[p]:
                    # get reference light curve
                    if gdat.booltpxf[p][o]:
                        arry, tsecrefr, tcam, tccd = miletos.read_tesskplr_file(listpathdownspoclcur[cntr], strgtypelcur='PDCSAP_FLUX')
                        gdat.refr.time[p][o][0] = arry[:, 0]
                        gdat.refr.rflx[p][o][0] = arry[:, 1]
                        gdat.refr.stdvrflx[p][o][0] = arry[:, 2]
                        gdat.boolrefrtser[p][o] = True
                        cntr += 1
                    else:
                        gdat.boolrefrtser[p][o] = False
    else:
        for p in gdat.indxinst:
            for o in gdat.indxtsec[p]:
                gdat.boolrefrtser[p][o] = True

    gdat.refr.indxpnts = [[[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in np.arange(gdat.refr.numbcatl)]
    
    gdat.listqualdata = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.indxtime = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    
    
    # one-dimensional coordinates of the images for each instrument
    gdat.xposimagodim = [[] for p in gdat.indxinst]
    gdat.yposimagodim = [[] for p in gdat.indxinst]
    
    # meshed coordinates of the images for each instrument
    gdat.xposimag = [[] for p in gdat.indxinst]
    gdat.yposimag = [[] for p in gdat.indxinst]
    
    gdat.xposimagpsfn = [[] for p in gdat.indxinst]
    gdat.yposimagpsfn = [[] for p in gdat.indxinst]
    gdat.numbtime = [[] for p in gdat.indxinst]
    
    for p in gdat.indxinst:
        gdat.numbtime[p] = np.empty(gdat.numbpoin[p], dtype=int)
        
        # image is between -0.5 and gdat.numbside - 0.5
        gdat.xposimagodim[p] = np.linspace(0., float(gdat.numbside[p] - 1), gdat.numbside[p])
        gdat.yposimagodim[p] = gdat.xposimagodim[p]
        gdat.xposimag[p], gdat.yposimag[p] = np.meshgrid(gdat.xposimagodim[p], gdat.xposimagodim[p])
    
    if gdat.fitt.typepsfnshap == 'empi':
        gdat.intgpsfnfitt = 6
        gdat.numbsidepsfnfitt = gdat.numbside[p] - 2 * gdat.intgpsfnfitt
        gdat.numbparapsfnempi = gdat.numbsidepsfnfitt**2
        gdat.indxparapsfnempi = np.arange(gdat.numbparapsfnempi)
        for p in gdat.indxinst:
            gdat.xposimagpsfn[p] = gdat.xposimag[p][gdat.intgpsfnfitt:-gdat.intgpsfnfitt, gdat.intgpsfnfitt:-gdat.intgpsfnfitt]
            gdat.yposimagpsfn[p] = gdat.yposimag[p][gdat.intgpsfnfitt:-gdat.intgpsfnfitt, gdat.intgpsfnfitt:-gdat.intgpsfnfitt]
    
    # number of lygos quality flags
    gdat.numblyqf = 3
    gdat.indxlyqf = np.arange(gdat.numblyqf)
    
    gdat.refr.indxpntswthn = [[[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in gdat.refr.indxcatl]
    gdat.refr.indxpntswthnbrgt = [[[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst] for q in gdat.refr.indxcatl]
    # get the WCS object for each sector
    gdat.listhdundata = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.listobjtwcss = [[[] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    
    gdat.listindxtimegoodlyqf = [[[[] for e in gdat.indxanls] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.fitt.arrytser = [[[[[[] for y in gdat.indxoffs] for x in gdat.indxoffs] for e in gdat.indxanls] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.fitt.arrytserinit = [[[[[[] for y in gdat.indxoffs] for x in gdat.indxoffs] for e in gdat.indxanls] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    gdat.listnois = [[[[] for e in gdat.indxanls] for o in gdat.indxtsec[p]] for p in gdat.indxinst]
    
    if gdat.strgtarg == 'Earth':
        
        import imageio

        pathepic = os.environ['LYGOS_DATA_PATH'] + '/data/epic_1b_20221020180856.png'
        print('Reading from %s...' % pathepic)
        gdat.imagepic = np.mean(imageio.imread(pathepic).astype(float), 2)
        intg = 32
        gdat.imagepic = tdpy.rebn_arry(gdat.imagepic, (int(gdat.imagepic.shape[0] / intg), int(gdat.imagepic.shape[1] / intg)))

        gdat.xposepic = np.arange(gdat.imagepic.shape[0]).astype(float)
        gdat.yposepic = np.arange(gdat.imagepic.shape[1]).astype(float)
        
        gdat.xposepic -= np.mean(gdat.xposepic)
        gdat.yposepic -= np.mean(gdat.yposepic)
        
        gdat.xposepic *= 2.
        gdat.yposepic *= 2.

        gdat.xposepic, gdat.yposepic = np.meshgrid(gdat.xposepic, gdat.yposepic)
    
        indx = np.where(gdat.imagepic > 0)
        print('indx[0]')
        summgene(indx[0])
        gdat.imagepic /= np.mean(gdat.imagepic[indx])

        figr, axis = plt.subplots(gdat.sizefigrsing)
        print('gdat.imagepic')
        summgene(gdat.imagepic)
        axis.hist(np.arcsinh(gdat.imagepic.flatten()), 200)
        #axis.set_ylabel('%s' % strgyaxi)
        #axis.set_xlabel(gdat.labltime)
        axis.set_yscale('log')
        path = gdat.pathvisutarg + 'histepic.%s' % (gdat.typefileplot)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
            
        gdat.funcintpepic = scipy.interpolate.interp2d(gdat.xposepic, gdat.yposepic, gdat.imagepic, kind='cubic', fill_value=0.)
    
    gdat.dictoutp['listipnt'] = [[] for p in gdat.indxinst]
    for p in gdat.indxinst:
        
        print(gdat.liststrginst[p])
        print('Pixel size: %g arcseconds' % gdat.sizepixl[p])
        if gdat.liststrgtypedata[p].startswith('simu'):
            if gdat.liststrginst[p].startswith('LSST'):
                gdat.latiobvt = -30.24506
                gdat.longobvt = -70.74913
                gdat.strgtimeobvtyear = '2024-06-01'
                gdat.listdelttimeobvtyear = np.arange(0., 365.25 * 10., 0.25)
                gdat.heigobvt = 2663 # [meters]

                massairr = tdpy.calc_visitarg(gdat.rasctarg, gdat.decltarg, gdat.latiobvt, gdat.longobvt, gdat.strgtimeobvtyear, \
                                                                                                                gdat.listdelttimeobvtyear, gdat.heigobvt)
                probobsv = 1. / massairr
                gdat.indxdayschun = np.choice(indxdays, p=probobsv)
                gdat.numbpoin[p] = gdat.indxdayschun.size
                gdat.listtime[p][o] = 2458119.5 + gdat.indxdayschun[o] + np.random.rand(2) * 0.5
        for o in gdat.indxtsec[p]:
            strgchun = retr_strgchun(gdat, p, o)
            updt_strgsave(gdat, strgchun, 1, 1, p, o)
            
            if not gdat.boolsimutargsynt:
                print(gdat.listlablpoin[p][o])
                if gdat.liststrginst[p] == 'TESS':
                    print('Camera: %d' % gdat.listtcam[o])
                    print('CCD: %d' % gdat.listtccd[o])
            
            if not gdat.boolsimutargsynt:
                if gdat.booltpxf[p][o]:
                    print('TPF data')
                else:
                    print('FFI data')
            
            gdat.liststrgmodl = ['fitt']
            if gdat.liststrgtypedata[p].startswith('simu'):
                gdat.liststrgmodl += ['true']
            
            if gdat.boolplotcntp or gdat.boolplotrflx or gdat.boolanim:
                if gdat.boolsimutargsynt or gdat.liststrginst[p] != 'TESS':
                    gdat.strgtitlcntpplot = '%s, %s, %s' % (gdat.labltarg, gdat.liststrginst[p], gdat.listlablpoin[p][o])
                else:
                    gdat.strgtitlcntpplot = '%s, %s, Sector %d, Cam %d, CCD %d' % (gdat.labltarg, gdat.liststrginst[p], \
                                                                                               gdat.listipnt[p][o], gdat.listtcam[o], gdat.listtccd[o])
            
            if gdat.boolsimutargsynt or gdat.liststrgtypedata[p] == 'simutargpartsynt' or gdat.liststrgtypedata[p] == 'simutargpartinje':
                if gdat.true.typepsfnshap == 'gauscirc':
                    gdat.true.parapsfn = np.array([gdat.true.sigmpsfn])
                if gdat.true.typepsfnshap == 'gauselli':
                    gdat.true.parapsfn = np.array([gdat.true.sigmpsfnxpos, gdat.true.sigmpsfnypos, gdat.true.fracskewpsfnxpos, gdat.true.fracskewpsfnypos])
            
            if gdat.booldiag:
                if gdat.typecade[p][o] is not None and gdat.timeexpo[p][o] is not None:
                    raise Exception('')

            if gdat.boolsimutargsynt or gdat.liststrgtypedata[p] == 'simutargpartsynt':
                if gdat.liststrginst[p] == 'TESS' or gdat.liststrginst[p].startswith('TGEO') or gdat.liststrginst[p] == 'ULTRASAT' or gdat.liststrginst[p] == 'TESSCam':
                    if gdat.typecade[p][o] == '2min':
                        gdat.timeexpo[p][o] = 2. / 60. / 24. # [days]
                    elif gdat.typecade[p][o] == '10mn':
                        gdat.timeexpo[p][o] = 10. / 60. / 24. # [days]
                    elif gdat.typecade[p][o] == '30mn':
                        gdat.timeexpo[p][o] = 30. / 60. / 24. # [days]
                    elif gdat.typecade[p][o] == '20sc':
                        gdat.timeexpo[p][o] = 20. / 60. / 60. / 24. # [days]
                    elif gdat.typecade[p][o] == '200s':
                        gdat.timeexpo[p][o] = 200. / 60. / 60. / 24. # [days]
                    else:
                        print('gdat.typecade[p][o]')
                        print(gdat.typecade[p][o])
                        raise Exception('')
                    # account for overhead
                    gdat.timeexpo[p][o] *= 0.8
                    
                if len(gdat.listtime[p][o]) == 0:
                    if gdat.liststrginst[p].startswith('LSST'):
                        gdat.listtime[p][o] = 2458119.5 + gdat.indxdayschun[o] + np.random.rand(2) * 0.5
                    else:
                        gdat.listtime[p][o] = 2458119.5 + np.arange(0., 1. / 24., gdat.timeexpo[p][o])
                
            # load the list of WCS objects for each pointing
            if (gdat.liststrgtypedata[p] == 'simutargpartinje' or gdat.liststrgtypedata[p] == 'obsd') and \
                                                 (gdat.booltesspast[o] or gdat.liststrginst[p] == 'LSST'):
            
                if gdat.liststrginst[p] == 'TESS':
                    if gdat.booltpxf[p][o]:
                        gdat.listhdundata[p][o] = gdat.listhdundataspoc[np.where(gdat.listipnt[p][o] == gdat.listtsecspoc)[0][0]]
                    else:
                        gdat.listhdundata[p][o] = gdat.listhdundataffimtess[np.where(gdat.listipnt[p][o] == gdat.listtsecffim)[0][0]]
                else:
                    if gdat.booldiag:
                        if isinstance(gdat.listhdundataffimtess[o], np.ndarray):
                            raise Exception('')
                    
                    # to be fixed for LSST
                    #gdat.listhdundata[p][o] = gdat.listhdundataffimothr[np.where(gdat.listipnt[p][o] == gdat.listtsecffim[p])[0][0]]
                
                if gdat.booldiag:
                    
                    if len(gdat.listhdundata[p][o]) == 0:
                        print('')
                        print('')
                        print('')
                        raise Exception('len(gdat.listhdundata[p][o]) == 0')
                        print(len(gdat.listhdundata[p][o]) == 0)
                    
                    if isinstance(gdat.listhdundata[p][o], np.ndarray):
                        print('')
                        print('')
                        print('')
                        raise Exception('')

                gdat.listobjtwcss[p][o] = astropy.wcs.WCS(gdat.listhdundata[p][o][2].header)
                
            # either simulate or read times and count maps
            if gdat.boolsimutargsynt or gdat.liststrgtypedata[p] == 'simutargpartsynt':
                
                pathfitssimu = gdat.pathdatatarg + 'cntpdata_%s_%s_%s.fits' % (gdat.liststrgtypedata[p], gdat.liststrginst[p], strgchun)
            
                if not os.path.exists(pathfitssimu):
                    # Create a new WCS object.  The number of axes must be set
                    print('Constructing a WCS object with pointing at RA, DEC = (%g, %g)...' % (gdat.rasctarg, gdat.decltarg))
                    objtwcss = astropy.wcs.WCS(naxis=2)
                    objtwcss.wcs.crpix = 0.5 * np.array([gdat.numbside[p], gdat.numbside[p]])
                    objtwcss.wcs.cdelt = gdat.sizepixl[p] * np.array([-1., 1.]) / 3600.
                    objtwcss.wcs.crval = [gdat.rasctarg, gdat.decltarg]
                    objtwcss.wcs.ctype = ["RA---AIR", "DEC--AIR"]
                    objtwcss.wcs.cunit = ['deg', 'deg']
                    #objtwcss.wcs.set_pv([(2, 1, 45.0)])
                    
                    gdat.listobjtwcss[p][o] = objtwcss

                else:
                    print('Reading from %s...' % pathfitssimu)
                    listhdun = astropy.io.fits.open(pathfitssimu)
                    gdat.listobjtwcss[p][o] = astropy.wcs.WCS(listhdun[0].header)
                    gdat.cntpdata = listhdun[1].data
            
            ## reference catalogs
            for q in gdat.refr.indxcatl:
                
                print('gdat.liststrgtypedata[p]')
                print(gdat.liststrgtypedata[p])
                if not gdat.boolsimutargsynt:
                    gdat.refr.cequ[q][p][o] = np.empty((gdat.refr.catl[q][p][o]['rasc'].size, 2))
                    gdat.refr.cequ[q][p][o][:, 0] = gdat.refr.catl[q][p][o]['rasc']
                    gdat.refr.cequ[q][p][o][:, 1] = gdat.refr.catl[q][p][o]['decl']
                    gdat.refr.cpix = gdat.listobjtwcss[p][o].all_world2pix(gdat.refr.cequ[q][p][o], 0)
                    gdat.refr.catl[q][p][o]['xpos'] = gdat.refr.cpix[:, 0]
                    gdat.refr.catl[q][p][o]['ypos'] = gdat.refr.cpix[:, 1]
                    if gdat.booltpxf[p][o] and gdat.numbside[p] < 11:
                        gdat.refr.catl[q][p][o]['xpos'] -= intg
                        gdat.refr.catl[q][p][o]['ypos'] -= intg
                    
            if gdat.boolsimutargsynt or gdat.liststrgtypedata[p] == 'simutargpartsynt':
                if not os.path.exists(pathfitssimu):
                    print('Simulating the images...')
                    # initialize the simulated data
                    gdat.cntpmodlsimu = np.zeros((gdat.numbside[p], gdat.numbside[p], gdat.listtime[p][o].size))
                    
                    ## add spatially unresolved sky background
                    if gdat.liststrginst[p] == 'ULTRASAT':
                        gdat.cntpmodlsimu += 10. # [e-/s]

                    if gdat.liststrginst[p].startswith('TGEO'):
                        gdat.cntpmodlsimu += 80. # [e-/s]
                    
                    if gdat.liststrginst[p] == 'TESSCam' or gdat.liststrginst[p] == 'TESS':
                        gdat.cntpmodlsimu += 100. # [e-/s]
                    
                    ## add sources
                    print('gdat.refr.catl[q][p]')
                    print(gdat.refr.catl[q][p])
                    print('gdat.refr.catl[q][p][o]')
                    print(gdat.refr.catl[q][p][o])
                    xpos = gdat.refr.catl[q][p][o]['xpos'][gdat.refr.indxpntswthn[q][p][o]]
                    ypos = gdat.refr.catl[q][p][o]['ypos'][gdat.refr.indxpntswthn[q][p][o]]
                    if gdat.refr.lablcatl[q] == 'TIC':
                        typesour = 'pnts'
                        cnts = gdat.refr.catl[q][p][o]['cntsesti'][gdat.refr.indxpntswthn[q][p][o]]
                    else:
                        cnts = np.zeros_like(xpos) + 1e8
                        typesour = 'Earth'
                    for t in np.arange(gdat.listtime[p][o].size):
                        if gdat.refr.lablcatl[q] == 'TIC' or gdat.refr.lablcatl[q] == 'CustDSCVR' and gdat.nametarg == 'Earth' and \
                                                        (gdat.liststrginst[p].startswith('TGEO') or gdat.liststrginst[p] == 'TESSCam' or gdat.liststrginst[p] == 'TESS'):
                            # Boolean flag indicating if the target is a Solar System object
                            if gdat.true.booltargssob:
                                
                                ampl = np.arctan(1. / gdat.true.disttarg) * 3600. / np.pi * 180.
                                deltrpos = np.sin(2. * np.pi * gdat.listtime[p][o][t])
                                deltxpos = deltrpos * np.cos(gdat.true.anglvelotarg)
                                deltypos = deltrpos * np.sin(gdat.true.anglvelotarg)
                                xpos += deltxpos
                                ypos += deltypos
                                #from astroquery.jplhorizons import Horizons
                                #
                                #statue_of_liberty = {'lon': -74.0466891, 'lat': 40.6892534, 'elevation': 0.093}
                                #objt = Horizons(id="Ceres", \
                                #                epochs=2458133.33546, \
                                #                #epochs=gdat.listtime[p][o][t], \
                                #                location=statue_of_liberty
                                #               )
                                #print('objt')
                                #print(objt)
                                #print('objt.ephemerides()')
                                #print(objt.ephemerides())
                                #print('objt.ephemerides().RA')
                                #print(objt.ephemerides().RA)
                                #raise Exception('')
                            
                            gdat.cntpmodlsimu[:, :, t] = retr_cntpmodl(gdat, p, 'true', xpos, ypos, cnts, 0., gdat.true.parapsfn, typesour=typesour)
                    
                    ## add dark current
                    gdat.cntpmodlsimu += gdat.timeexpo[p][o] * 0.1 # [e-/s]
                    
                    ## draw a Poisson realization of the photon count per pixel
                    gdat.cntpdatasimu = np.random.poisson(gdat.cntpmodlsimu).astype(float)
                    
                    # add bias
                    gdat.cntpdatasimu += 1000.
                    
                    # add readout noise
                    gdat.cntpdatasimu += np.round(8. * np.random.randn(gdat.cntpdatasimu.size).reshape(gdat.cntpdatasimu.shape))
                    
                    ## change from photoelectrons to ADUs
                    gdat.cntpmodlsimu *= gdat.gainphot[p]
                    gdat.cntpdata = np.round(gdat.cntpmodlsimu)
                    
                    # write the simulated data to disk
                    #objthead = astropy.io.fits.Header()
                    # construct a FITS header from the WCS object
                    objthead = gdat.listobjtwcss[p][o].to_header()
                    objthead['OBSERVER'] = 'Tansu Daylan'
                    objthead['COMMENT'] = 'Simulated using lygos'
                    hdunprim = astropy.io.fits.PrimaryHDU(header=objthead)
                    hdunimag = astropy.io.fits.ImageHDU(gdat.cntpdata)
                    listhdun = astropy.io.fits.HDUList([hdunprim, hdunimag])
                    print('heeey')
                    print('gdat.cntpdata')
                    summgene(gdat.cntpdata)
                    print('Writing to %s...' % pathfitssimu)
                    listhdun.writeto(pathfitssimu)

            elif gdat.liststrgtypedata[p] == 'simutargpartinje' or gdat.liststrgtypedata[p] == 'obsd':
                
                # get data
                ## read the FITS files
                #print(gdat.listhdundata[p][o][1].data.names)
                # times
                if gdat.booldiag:
                    if len(gdat.listhdundata[p][o]) == 0:
                        print('')
                        print('')
                        print('')
                        print('po')
                        print(p, o)
                        print('gdat.listhdundata[p][o]')
                        summgene(gdat.listhdundata[p][o])
                        raise Exception('gdat.listhdundata[p][o] is empty.')
                    
                gdat.listtime[p][o] = gdat.listhdundata[p][o][1].data['TIME'] + 2457000
                
                # quality flag
                gdat.listqualdata[p][o] = gdat.listhdundata[p][o][1].data['QUALITY']
                
                ## count per pixel
                gdat.cntpdata = (gdat.listhdundata[p][o][1].data['FLUX'] + \
                                                            gdat.listhdundata[p][o][1].data['FLUX_BKG']).swapaxes(0, 2).swapaxes(0, 1)
                
                if gdat.booltpxf[p][o]:
                    #indxtsectemp = np.where(gdat.listipnt[p][o] == gdat.listtsecspoc)[0][0]
                    #gdat.listtime[p][o] = gdat.listtime[p][o][gdat.indxtimegoodspoc[indxtsectemp]]
                    #gdat.cntpdata = gdat.cntpdata[:, :, gdat.indxtimegoodspoc[indxtsectemp]]

                    if gdat.numbside[p] < 11:
                        intg = int((11 - gdat.numbside[p]) / 2)
                        gdat.cntpdata = gdat.cntpdata[intg:11-intg, intg:11-intg, :]

                print('Number of raw data points: %d' % gdat.listtime[p][o].size)
                
                if gdat.booltpxf[p][o]:
                    gdat.numbside[p] = gdat.cntpdata.shape[1]
                
                #booldatagood = np.isfinite(gdat.listtime[p][o])
                print('Checking to mask out bad data with zero or negative counts...')
                booldatagood = np.any(gdat.cntpdata > 0, axis=(0, 1))
                if gdat.boolmaskqual:
                    print('Checking to mask out bad data with quality flags...')
                    booldatagood = booldatagood & (gdat.listqualdata[p][o] == 0)

                    if limttimeignoqual is not None:
                        print('Ignoring the quality mask between %g and %g...' % (limttimeignoqual[0], limttimeignoqual[1]))
                        booldatagood = booldatagood & ((limttimeignoqual[0] < gdat.listtime[p][o]) & (gdat.listtime[p][o] < limttimeignoqual[1]))
                indxtimedatagood = np.where(booldatagood)[0]
                fracgood = 100. * float(indxtimedatagood.size) / gdat.listtime[p][o].size
                print('Fraction of unmasked (good) times: %.4g percent' % fracgood)
                if indxtimedatagood.size == 0:
                    print('No good data found for this sector. The returned list will have an empty element.')
                    continue
        
                # keep good times and discard others
                gdat.listtime[p][o] = gdat.listtime[p][o][indxtimedatagood]
                gdat.cntpdata = gdat.cntpdata[:, :, indxtimedatagood]
            
            else:   
                raise Exception('')

            if gdat.strgtarg == 'Earth':
                xposeart = gdat.numbside[p] / 2.
                yposeart = xposeart
                gdat.funcepic = gdat.funcintpepic(gdat.xposimagodim[p] - xposeart, gdat.yposimagodim[p] - yposeart).reshape(gdat.xposimag[p].shape)
                
                gdat.funcepic -= np.amin(gdat.funcepic)
                #gdat.funcepic[gdat.funcepic < 0] = 0.

                if (gdat.funcepic < 0.).any():
                    print('gdat.funcepic')
                    summgene(gdat.funcepic)
                    raise Exception('')

            for q in gdat.refr.indxcatl:
                gdat.refr.numbpnts[q][p][o] = gdat.refr.catl[q][p][o]['xpos'].size
                gdat.refr.indxpnts[q][p][o] = np.arange(gdat.refr.numbpnts[q][p][o])
                
                ## indices of the reference catalog sources within the cutout
                gdat.refr.indxpntswthn[q][p][o] = np.where((gdat.refr.catl[q][p][o]['xpos'] > -0.5) & (gdat.refr.catl[q][p][o]['xpos'] < gdat.numbside[p] - 0.5) & \
                                                (gdat.refr.catl[q][p][o]['ypos'] > -0.5) & (gdat.refr.catl[q][p][o]['ypos'] < gdat.numbside[p] - 0.5))[0]

                
                print('Number of reference sources inside the cutout is %d...' % gdat.refr.indxpntswthn[q][p][o].size)
                gdat.refr.indxpntswthnbrgt[q][p][o] = np.intersect1d(gdat.refr.indxpntswthn[q][p][o], gdat.indxpntsbrgt[q])
                print('Number of reference sources that are bright & blended and inside the cutout is %d...' % gdat.refr.indxpntswthnbrgt[q][p][o].size)
                
                if gdat.booldiag:
                    if len(gdat.refr.indxpntswthnbrgt[0][p][o]) == 0:
                        print('')
                        print('')
                        print('')
                        print('gdat.refr.indxpntswthn[0][p][o]')
                        print(gdat.refr.indxpntswthn[0][p][o])
                        print('gdat.indxpntsbrgt[0]')
                        print(gdat.indxpntsbrgt[0])
                        print('gdat.refr.indxpntswthnbrgt[0][p][o]')
                        print(gdat.refr.indxpntswthnbrgt[0][p][o])
                        raise Exception('len(gdat.refr.indxpntswthnbrgt[0][p][o]) == 0')
            
            gdat.numbtime[p][o] = gdat.listtime[p][o].size
            gdat.indxtime[p][o] = np.arange(gdat.numbtime[p][o])

            if gdat.booldiag:
                if gdat.cntpdata.shape[2] != gdat.numbtime[p][o]:
                    print('')
                    print('')
                    print('')
                    print('gdat.numbtime[p][o]')
                    print(gdat.numbtime[p][o])
                    print('gdat.cntpdata')
                    summgene(gdat.cntpdata)
                    raise Exception('Last dimension of cntpdata should be same as gdat.numbtime[p][o].')
                                
            gdat.indxside = np.arange(gdat.numbside[p])

            # set up time windows
            ## number of time windows
            if gdat.listlimttimetzom is not None:
                gdat.numblimttime = len(gdat.listlimttimetzom)
            else:
                gdat.numblimttime = 1
            gdat.indxlimttime = np.arange(gdat.numblimttime)
            gdat.indxtimelimt = [[] for ik in gdat.indxlimttime]
            ## indices for each time window
            for ik in gdat.indxlimttime:
                if gdat.listlimttimetzom is None:
                    gdat.indxtimelimt[ik] = gdat.indxtime
                else:
                    gdat.indxtimelimt[ik] = np.where((gdat.listtime[p][o] > gdat.listlimttimetzom[ik][0]) & (gdat.listtime[p][o] < gdat.listlimttimetzom[ik][1]))[0]

            for q in gdat.refr.indxcatl:
                gdat.refr.catlbase[q]['xpostime'] = np.zeros((gdat.numbtime[p][o], gdat.refr.numbpntsbase[q])) + gdat.refr.catl[q][p][o]['xpos'][None, :]
                gdat.refr.catlbase[q]['ypostime'] = np.zeros((gdat.numbtime[p][o], gdat.refr.numbpntsbase[q])) + gdat.refr.catl[q][p][o]['ypos'][None, :]
                    
            if gdat.booldiag:
                for name in ['xpos', 'ypos']:
                    if not np.isfinite(gdat.refr.catl[q][p][o][name]).all():
                        print('name')
                        print(name)
                        print('gdat.refr.catl[q][p][o][name]')
                        summgene(gdat.refr.catl[q][p][o][name])
                        print(gdat.refr.catl[q][p][o][name])
                        raise Exception('')

            #raise Exception('')

            if gdat.liststrgtypedata[p] == 'simutargpartinje':
                
                # generate data
                gdat.cntpmodlsimu = np.empty((gdat.numbside[p], gdat.numbside[p], gdat.numbtime[p][o]))
            
                for t in np.arange(gdat.numbtime[p][o]):
                    gdat.cntpmodlsimu[:, :, t] = retr_cntpmodl(gdat, p, 'true', gdat.refr.catl[q][p][o]['xpos'], \
                                                               gdat.refr.catl[q][p][o]['ypos'], gdat.refr.catl[q][p][o]['cnts'], 0., gdat.true.parapsfn, typesour='pnts')
                gdat.cntpdata += gdat.cntpmodlsimu

            if gdat.booldiag:
                if gdat.cntpdata.shape[2] != gdat.numbtime[p][o]:
                    print('')
                    print('')
                    print('')
                    print('gdat.numbtime[p][o]')
                    print(gdat.numbtime[p][o])
                    print('gdat.cntpdata')
                    summgene(gdat.cntpdata)
                    raise Exception('Last dimension of cntpdata should be same as gdat.numbtime[p][o].')
                                
            gdat.cntpdatasexp = gdat.cntpdata[:, :, 0]
        
            gdat.cntpdatatmed = np.nanmedian(gdat.cntpdata, axis=-1)
            gdat.cntpdatatser = np.median(np.median(gdat.cntpdata, axis=0), axis=0)
            
            if not np.isfinite(gdat.cntpdatatser).any():
                print('No time stamp without NaNs are found. Skipping this sector...')
                continue

            if gdat.fitt.typepsfnshap == 'data':
                gdat.cntpdatapsfn = gdat.cntpdatatmed - np.percentile(gdat.cntpdatatmed, 90.)
                gdat.cntpdatapsfn[np.where(gdat.cntpdatapsfn < 0)] = 0
                gdat.cntpdatapsfn /= np.mean(gdat.cntpdatapsfn)
                #gdat.funcintppsfn = scipy.interpolate.interp2d(gdat.xposimagpsfn, gdat.yposimagpsfn, gdat.cntpdatapsfn, \
                #                                      kind='cubic', fill_value=0.)(gdat.xposimag[p], gdat.yposimag[p])
        
            if 'aper' in gdat.listnameanls:
                if gdat.listpixlapertarg is None:
                    gdat.listpixlapertarg = [[]]
                    gdat.listpixlapertarg[0] = np.where(gdat.cntpdatatmed > np.percentile(gdat.cntpdata, 60.))
                    gdat.listpixlaperback = [[]]
                    gdat.listpixlaperback[0] = np.where(gdat.cntpdatatmed < np.percentile(gdat.cntpdata, 60.))
                gdat.numbaper = len(gdat.listpixlapertarg)
                gdat.indxaper = np.arange(gdat.numbaper)
                                
            if len(gdat.cntpdata) == 0:
                raise Exception('')

            if not np.isfinite(gdat.cntpdata).all():
                print('gdat.cntpdata')
                summgene(gdat.cntpdata)
                print('Not all counts are finite!')
                #raise Exception('')

            if gdat.booldiag:
                if gdat.cntpdata.shape[2] != gdat.numbtime[p][o]:
                    print('')
                    print('')
                    print('')
                    print('gdat.cntpdata')
                    summgene(gdat.cntpdata)
                    raise Exception('Last dimension of cntpdata should be same as gdat.numbtime[p][o].')
                                
            if gdat.booldiag:
                if gdat.cntpdatatmed.shape[0] != gdat.numbside[p]:
                    print('')
                    print('')
                    print('')
                    raise Exception('First dimension of cntpdatatmed should be same as numbside[p].')
            
            #if not np.isfinite(gdat.cntpdatatmed).all():
            #    raise Exception('')
            
            # plot data with initial catalogs (before PM correction)
            if gdat.boolplotcntp:
                strgtitl = gdat.strgtitlcntpplot
                for typecntpscal in gdat.listtypecntpscal:
                    plot_cntp(gdat, gdat.cntpdatasexp, p, o, typecntpscal, gdat.pathvisutargsexp, 'cntpdatasexp_nopm', 'refr', strgtitl=strgtitl)
                
            if not gdat.boolsimutargsynt:
                
                if gdat.booldiag:
                    for name in ['rasc', 'decl']:
                        if not np.isfinite(gdat.refr.catl[q][p][o][name]).all():
                            print('name')
                            print(name)
                            print('gdat.refr.catl[q][p][o][name]')
                            print(gdat.refr.catl[q][p][o][name])
                            summgene(gdat.refr.catl[q][p][o][name])
                            raise Exception('')
            
                print('Correcting the reference catalog for proper motion...')
                for q in gdat.refr.indxcatl:
                    # epoch for correcting the RA and DEC for proper motion.
                    gdat.epocpmot = (np.mean(gdat.listtime[p][o]) - 2433282.5) / 365.25 + 1950.
                    print('Epoch: %g' % gdat.epocpmot)
                    
                    pmra = gdat.refr.catlbase[q]['pmra']
                    pmde = gdat.refr.catlbase[q]['pmde']
                    rascorig = gdat.refr.catlbase[q]['rascorig']
                    declorig = gdat.refr.catlbase[q]['declorig']
                    
                    indx = np.where(np.isfinite(pmra) & np.isfinite(rascorig))[0]
                    gdat.refr.catl[q][p][o]['rasc'][indx] = gdat.refr.catlbase[q]['rascorig'][indx] + \
                                                                gdat.refr.catlbase[q]['pmra'][indx] * (gdat.epocpmot - 2015.5) / (1000. * 3600.)
                    
                    if not np.isfinite(gdat.refr.catl[q][p][o]['rasc']).all():
                        raise Exception('')

                    indx = np.where(np.isfinite(pmde) & np.isfinite(declorig))[0]
                    gdat.refr.catl[q][p][o]['decl'][indx] = gdat.refr.catlbase[q]['declorig'][indx] + \
                                                                gdat.refr.catlbase[q]['pmde'][indx] * (gdat.epocpmot - 2015.5) / (1000. * 3600.)
                        
                    if not np.isfinite(gdat.refr.catl[q][p][o]['decl']).all():
                        raise Exception('')

                    gdat.refr.cequ[q][p][o] = np.empty((gdat.refr.catl[q][p][o]['rasc'].size, 2))
                    gdat.refr.cequ[q][p][o][:, 0] = gdat.refr.catl[q][p][o]['rasc']
                    gdat.refr.cequ[q][p][o][:, 1] = gdat.refr.catl[q][p][o]['decl']
                    gdat.refr.cpix = gdat.listobjtwcss[p][o].all_world2pix(gdat.refr.cequ[q][p][o], 0)
                    gdat.refr.catl[q][p][o]['xpos'] = gdat.refr.cpix[:, 0]
                    gdat.refr.catl[q][p][o]['ypos'] = gdat.refr.cpix[:, 1]
                    
                    if not np.isfinite(gdat.refr.catl[q][p][o]['xpos']).all():
                        print('gdat.refr.catl[q][p][o][xpos]')
                        print(gdat.refr.catl[q][p][o]['xpos'])
                        summgene(gdat.refr.catl[q][p][o]['xpos'])
                        raise Exception('')

                    if not np.isfinite(gdat.refr.catl[q][p][o]['ypos']).all():
                        raise Exception('')

                    if gdat.booltpxf[p][o] and gdat.numbside[p] < 11:
                        gdat.refr.catl[q][p][o]['xpos'] -= intg
                        gdat.refr.catl[q][p][o]['ypos'] -= intg
            
                if gdat.booldiag:
                    for name in ['rasc', 'decl']:
                        if not np.isfinite(gdat.refr.catl[q][p][o][name]).all():
                            print('name')
                            print(name)
                            print('gdat.refr.catl[q][p][o][name]')
                            print(gdat.refr.catl[q][p][o][name])
                            summgene(gdat.refr.catl[q][p][o][name])
                            raise Exception('')
            
            if gdat.timeoffs != 0.:
                gdat.labltime = 'Time [BJD - %d]' % gdat.timeoffs
            else:
                gdat.labltime = 'Time [BJD]'
            
            if gdat.liststrginst[p] == 'TESS':
                gdat.pathcbvs = gdat.pathdatalygo + 'cbvs/'
                if gdat.booldetrcbvs:
                    path = gdat.pathcbvs + \
                             fnmatch.filter(os.listdir(gdat.pathcbvs), 'tess*-s%04d-%d-%d-*-s_cbv.fits' % (gdat.listipnt[p][o], gdat.listtcam[o], gdat.listtccd[o]))[0]
                    print('Reading from %s...' % path)
                    listhdun = astropy.io.fits.open(path)
                    #listhdun.info()
                    timecbvs = listhdun[1].data['TIME']
                    gdat.numbcbvs = 5
                    gdat.indxcbvs = np.arange(gdat.numbcbvs)
                    timecbvstemp = listhdun[1].data['TIME'] + 2457000
                    cbvsraww = np.empty((timecbvstemp.size, gdat.numbcbvs))
                    gdat.cbvs = np.empty((gdat.numbtime[p][o], gdat.numbcbvs))
                    for i in gdat.indxcbvs:
                        cbvsraww[:, i] = listhdun[1].data['VECTOR_%i' % (i + 1)]
                        gdat.cbvs[:, i] = scipy.interpolate.interp1d(timecbvstemp, cbvsraww[:, i])(gdat.listtime[p][o])
                    
                    gdat.cbvstmpt = np.ones((gdat.numbtime[p][o], gdat.numbcbvs + 1))
                    gdat.cbvstmpt[:, :-1] = gdat.cbvs
                
            if gdat.boolplotcent:
                # plot centroid
                for a in range(1):
                    if a == 0:
                        numbplot = 2
                    else:
                        numbplot = gdat.numbcbvs

                    for k in range(numbplot):
                        figr, axis = plt.subplots(gdat.sizefigrdoublcur)
                        
                        if a == 0:
                            if k == 0:
                                strgyaxi = 'x'
                                posi = gdat.xposimag[p]
                            else:
                                strgyaxi = 'y'
                                posi = gdat.yposimag[p]
                            strgplot = 'cent'
                            temp = np.sum(posi[None, :, :, None] * gdat.cntpdata, axis=(0, 1, 2)) / np.sum(gdat.cntpdata, axis=(0, 1, 2))
                        else:
                            temp = gdat.cbvs[:, k]
                            strgyaxi = 'CBV$_{%d}$' % k
                            posi = gdat.xposimag[p]
                            strgplot = 'cbvs'
                        axis.plot(gdat.listtime[p][o] - gdat.timeoffs, temp, ls='', marker='.', ms=1)
                        axis.set_ylabel('%s' % strgyaxi)
                        axis.set_xlabel(gdat.labltime)
                        path = gdat.pathvisutarg + '%s_%s_%02d.%s' % (strgplot, strgchun, k, gdat.typefileplot)
                        print('Writing to %s...' % path)
                        plt.savefig(path)
                        plt.close()
            
            gdat.quat = None
            if gdat.liststrginst[p] == 'TESS' and gdat.listipnt[p][o] < gdat.tseccurr:
                gdat.quat = np.zeros((gdat.numbtime[p][o], 3))
                #if len(os.listdir(gdat.pathcbvs)) == 0 and gdat.boolplotquat:
                if gdat.boolplotquat:
                    print('Reading quaternions...')
                    pathquatbase = gdat.pathdatalygo + 'quat/'
                    listfile = fnmatch.filter(os.listdir(pathquatbase), 'tess*_sector%02d-quat.fits' % gdat.listipnt[p][o])
                    if len(listfile) > 0:
                        path = gdat.pathvisutarg + 'tserquat_sc%02d.%s' % (gdat.listipnt[p][o], gdat.typefileplot)
                        if not os.path.exists(path):
                            pathquat = pathquatbase + listfile[0]
                            listhdun = astropy.io.fits.open(pathquat)
                            dataquat = listhdun[gdat.listtcam[o]].data
                            headquat = listhdun[gdat.listtcam[o]].header
                            #for k, key in enumerate(headquat.keys()):
                            #    print(key + ' ' + str(headquat[k]))
                            figr, axis = plt.subplots(3, 1, figsize=gdat.sizefigrdoublcur, sharex=True)
                            for k in range(1, 4):
                                strg = 'C%d_Q%d' % (gdat.listtcam[o], k)
                                gdat.quat[:, k-1] = scipy.interpolate.interp1d(dataquat['TIME'] + 2457000, dataquat[strg], \
                                                                                                fill_value=0, bounds_error=False)(gdat.listtime[p][o])
                                minm = np.percentile(dataquat[strg], 0.05)
                                maxm = np.percentile(dataquat[strg], 99.95)
                                #axis[k-1].plot(dataquat['TIME'] + 2457000 - gdat.timeoffs, dataquat[strg], ls='', marker='.', ms=1)
                                axis[k-1].plot(gdat.listtime[p][o] - gdat.timeoffs, gdat.quat[:, k-1], ls='', marker='.', ms=1)
                                axis[k-1].set_ylim([minm, maxm])
                                axis[k-1].set_ylabel('$Q_{%d}$' % k)
                            axis[2].set_xlabel(gdat.labltime)
                            print('Writing to %s...' % path)
                            plt.savefig(path)
                            plt.close()
                    else:
                        print('Could not find any quaternion file.')

            if not np.isfinite(gdat.cntpdata).all():
                print('Warning! Not all data are finite.')
            
            if gdat.catlextr is not None:
                for qq in gdat.refr.indxcatl:
                    gdat.refr.catl[qq+1]['rasc'] = gdat.catlextr[qq]['rasc']
                    gdat.refr.catl[qq+1]['decl'] = gdat.catlextr[qq]['decl']

            if np.amin(gdat.cntpdata) < 0:
                print('Minimum of the image is negative.')

            # data variance
            gdat.vari = np.copy(gdat.cntpdata)

            
            #raise Exception('')

            if not gdat.boolfitt:
                continue

            gdat.fitt.catl = {}
        
            ## fitting catalog
            if gdat.boolsimutargsynt:
                for name in ['xpos', 'ypos', 'cnts', 'labl']:
                    gdat.fitt.catl[name] = gdat.refr.catl[0][p][o][name]
            if not gdat.boolsimutargsynt:
                
                # copy the first reference catalog to the fitting catalog
                for strgfeat in gdat.refr.liststrgfeat[q]:
                    gdat.fitt.catl[strgfeat] = gdat.refr.catl[0][p][o][strgfeat][gdat.refr.indxpntswthnbrgt[0][p][o]]
                
                if gdat.booldiag:
                    if len(gdat.fitt.catl['labl']) != len(gdat.fitt.catl['rasc']):
                        print('gdat.refr.catl[0][p][o][rasc]')
                        print(gdat.refr.catl[0][p][o]['rasc'])
                        print('gdat.refr.catl[0][p][o][labl]')
                        print(gdat.refr.catl[0][p][o]['labl'])
                        print('gdat.fitt.catl[rasc]')
                        print(gdat.fitt.catl['rasc'])
                        print('gdat.fitt.catl[labl]')
                        print(gdat.fitt.catl['labl'])
                        raise Exception('')

                skyyfitttemp = np.empty((gdat.fitt.catl['rasc'].size, 2))
                skyyfitttemp[:, 0] = gdat.fitt.catl['rasc']
                skyyfitttemp[:, 1] = gdat.fitt.catl['decl']
                if gdat.fitt.catl['rasc'].size == 0:
                    print('There is no target in the fitting catalog.')
                    raise Exception('')
                # transform sky coordinates into dedector coordinates and filter
                posifitttemp = gdat.listobjtwcss[p][o].all_world2pix(skyyfitttemp, 0)
                gdat.fitt.catl['xpos'] = posifitttemp[:, 0]
                gdat.fitt.catl['ypos'] = posifitttemp[:, 1]
                if gdat.booltpxf[p][o] and gdat.numbside[p] < 11:
                    gdat.fitt.catl['xpos'] -= intg
                    gdat.fitt.catl['ypos'] -= intg
                
            if gdat.booldiag:
                if len(gdat.fitt.catl['labl']) != len(gdat.fitt.catl['xpos']):
                    print('gdat.fitt.catl[xpos]')
                    print(gdat.fitt.catl['xpos'])
                    print('gdat.fitt.catl[labl]')
                    print(gdat.fitt.catl['labl'])
                    raise Exception('')
        
            gdat.fitt.liststrgfeat = gdat.refr.liststrgfeat[q]
            
            if not gdat.boolsimutargsynt:
                if gdat.booldiag:
                    for name in ['rasc', 'decl']:
                        if not np.isfinite(gdat.fitt.catl[name]).all():
                            print('name')
                            print(name)
                            print('gdat.fitt.catl[name]')
                            print(gdat.fitt.catl[name])
                            summgene(gdat.fitt.catl[name])
                            raise Exception('')

                    for name in ['xpos', 'ypos']:
                        if not np.isfinite(gdat.fitt.catl[name]).all():
                            print('name')
                            print(name)
                            print('gdat.fitt.catl[name]')
                            print(gdat.fitt.catl[name])
                            summgene(gdat.fitt.catl[name])
                            raise Exception('')

            if gdat.boolmerg:
                print('Checking whether it is necessary to merge fitting source pairs that are too close...')
                while True:
                    
                    #print('Iteration %d...' % n)
                    
                    # for each source, find the distance to all other sources
                    dist = np.sqrt((gdat.fitt.catl['xpos'][:, None] - gdat.fitt.catl['xpos'][None, :])**2 + \
                                                                (gdat.fitt.catl['ypos'][:, None] - gdat.fitt.catl['ypos'][None, :])**2)
                    dist[range(dist.shape[0]), range(dist.shape[0])] = 1e10
                    # find the index of the closest neighbor
                    n, m = np.unravel_index(np.argmin(dist), dist.shape)
                    #n, m = np.argmin(dist)
                    
                    if dist[n, m] < 0.5:
                        
                        cnts = gdat.fitt.catl['cntsesti']
                        gdat.fitt.catl['xpos'][n] = (cnts[n] * gdat.fitt.catl['xpos'][n] + cnts[m] * gdat.fitt.catl['xpos'][m]) / (cnts[n] + cnts[m])
                        gdat.fitt.catl['ypos'][n] = (cnts[n] * gdat.fitt.catl['ypos'][n] + cnts[m] * gdat.fitt.catl['ypos'][m]) / (cnts[n] + cnts[m])
                        
                        # delete the close source
                        for name in gdat.fitt.liststrgfeat:
                            gdat.fitt.catl[name] = np.delete(gdat.fitt.catl[name], m)
                            
                        print('Merging model point sources...')
                        
                    else:
                        break
            
            if gdat.fitt.catl['xpos'].size == 0:
                print('No fitting source found...')
                print('')
                return gdat.dictoutp
            
            gdat.fitt.numbpnts = np.empty(gdat.numbanls, dtype=int)
            gdat.fitt.numbpnts[0] = gdat.fitt.catl['xpos'].size
            
            if gdat.booldiag:
                for e, nameanls in enumerate(gdat.listnameanls):
                    if len(gdat.fitt.catl['labl']) != gdat.fitt.numbpnts[e]:
                        print('')
                        print('')
                        print('')
                        print('gdat.fitt.catl[labl]')
                        print(gdat.fitt.catl['labl'])
                        print('gdat.fitt.numbpnts')
                        print(gdat.fitt.numbpnts)
                        print('gdat.fitt.numbpntsneig')
                        print(gdat.fitt.numbpntsneig)
                        print('gdat.fitt.numbcomp')
                        print(gdat.fitt.numbcomp)
                        print('gdat.listnameanls')
                        print(gdat.listnameanls)
                        raise Exception('length of gdat.fitt.catl[labl] should be same as gdat.fitt.numbpnts[e].')

            print('gdat.fitt.numbpnts')
            print(gdat.fitt.numbpnts)
            print('gdat.fitt.catl[xpos]')
            print(gdat.fitt.catl['xpos'])
        
            gdat.numbpixl = gdat.numbside[p]**2
            
            # types of image plots
            listnameplotcntp = ['cntpdata', 'cntpmodl', 'cntpresi']
            
            if not np.isfinite(gdat.cntpdata).all():
                print('There is NaN in the data!')
                print('gdat.cntpdata')
                summgene(gdat.cntpdata)
                print('gdat.cntpdatatmed')
                summgene(gdat.cntpdatatmed)
                print('np.median(np.median(gdat.cntpdata, axis=0), axis=0)')
                summgene(np.median(np.median(gdat.cntpdata, axis=0), axis=0))

            gdat.boolbackoffs = True
            gdat.boolposioffs = False
            
            gdat.cntpback = np.zeros_like(gdat.cntpdata)

            if np.amin(gdat.cntpdata) < 1.:
                print('Minimum of the data is not positive.')

            if not np.isfinite(gdat.cntpback).all():
                raise Exception('')
            
            if gdat.cntpdata.shape[0] != gdat.numbside[p]:
                raise Exception('')

            if gdat.booldiag:
                if gdat.cntpdata.shape[2] != gdat.numbtime[p][o]:
                    print('')
                    print('')
                    print('')
                    print('gdat.cntpdata')
                    summgene(gdat.cntpdata)
                    raise Exception('Last dimension of cntpdata should be same as gdat.numbtime[p][o].')
                                
            if gdat.cntpdata.shape[1] != gdat.numbside[p]:
                raise Exception('')

            gdat.fitt.numbpntsneig = np.empty(gdat.numbanls, dtype=int)
            gdat.fitt.indxpntsneig = [[] for e in gdat.indxanls]
            gdat.fitt.numbcomp = np.empty(gdat.numbanls, dtype=int)
            gdat.fitt.indxcomp = [[] for e in gdat.indxanls]
            gdat.fitt.indxpnts = [[] for e in gdat.indxanls]
            for e, nameanls in enumerate(gdat.listnameanls):
                
                if nameanls == 'aper':
                    gdat.fitt.numbpntsneig[e] = 0
                    gdat.fitt.numbcomp[e] = 2
                elif nameanls == 'psfn':
                    gdat.fitt.numbpntsneig[e] = gdat.fitt.numbpnts[e] - 1
                    gdat.fitt.numbcomp[e] = gdat.fitt.numbpnts[e] + 1
                else:
                    print('')
                    print('')
                    print('')
                    print('e')
                    print(e)
                    print('gdat.listnameanls')
                    print(gdat.listnameanls)
                    print('gdat.fitt.numbpntsneig[e]')
                    print(gdat.fitt.numbpntsneig[e])
                    raise Exception('nameanls is undefined.')
                
                gdat.fitt.indxpntsneig[e] = np.arange(gdat.fitt.numbpntsneig[e])
                gdat.fitt.indxpnts[e] = np.arange(gdat.fitt.numbpnts[e])
                gdat.fitt.indxcomp[e] = np.arange(gdat.fitt.numbcomp[e])

                gdat.strgheadtotl = 'time [BJD]'
                gdat.strgheadtotl += ', target rel flux, target rel flux err'
                for k in gdat.fitt.indxpntsneig[e]:
                    gdat.strgheadtotl += ', neig %d rel flux, neig %d rel flux err' % (k+1, k+1)
                gdat.strgheadtotl += ', bkg rel flux, bkg rel flux err'
            
            if gdat.booldiag:
                for e, nameanls in enumerate(gdat.listnameanls):
                    if len(gdat.fitt.catl['labl']) != gdat.fitt.numbpnts[e]:
                        print('')
                        print('')
                        print('')
                        print('gdat.fitt.catl[labl]')
                        print(gdat.fitt.catl['labl'])
                        print('gdat.fitt.numbpnts')
                        print(gdat.fitt.numbpnts)
                        print('gdat.fitt.numbpntsneig')
                        print(gdat.fitt.numbpntsneig)
                        print('gdat.fitt.numbcomp')
                        print(gdat.fitt.numbcomp)
                        print('gdat.listnameanls')
                        print(gdat.listnameanls)
                        raise Exception('length of gdat.fitt.catl[labl] should be same as gdat.fitt.numbpnts[e].')

            for strgmodl in gdat.liststrgmodl:
                gmod = getattr(gdat, strgmodl)
                
                if strgmodl == 'fitt':
                    catl = gdat.fitt.catl
                else:
                    catl = gdat.refr.catl
                lablcomp = [[] for k in gmod.indxcomp[e]]
                for k in gmod.indxcomp[e]:
                    if k == 0:
                        lablcomp[k] = gdat.labltarg
                        if gdat.labltarg != catl['labl'][k]:
                            lablcomp[k] += ' (%s)' % catl['labl'][k]
                    elif k == gmod.numbcomp - 1:
                        lablcomp[k] = 'Background'
                    else:
                        lablcomp[k] = '%s (Centered on %s)' % (catl['labl'][k], catl['labl'][0])
            tdpy.setp_para_defa(gdat, 'full', 'lablcomp', lablcomp)
    
            updt_strgsave(gdat, strgchun, 1, 1, p, o)
            
            if gdat.typepsfninfe == 'locl' or gdat.typepsfninfe == 'both':
                
                strgextn = 'psfn_%s_%s' % (strgchun, gdat.fitt.typepsfnshap)
               
                path = gdat.pathvisutargsexp + 'parapmed_%s.csv' % strgextn
                if not os.path.exists(path):
                    
                    print('Modeling the data with a floating PSF...')
                
                    # fit for the PSF
                    numbsampwalk = 100000
                    numbsampburnwalkinit = 0
                    numbsampburnwalk = int(0.9 * numbsampwalk)
                    
                    gdat.typefittpsfnposi = 'fixd'
                    gdat.typefittpsfncnts = 'vari'
                    
                    listnamepara = []
                    listlablpara = []
                    listminmpara = []
                    listmaxmpara = []
                    
                    cntr = 0
                    if gdat.typefittpsfnposi == 'vari':
                        gdat.indxparapsfnxpos = np.empty(gdat.fitt.numbpnts, dtype=int)
                        gdat.indxparapsfnypos = np.empty(gdat.fitt.numbpnts, dtype=int)
                        for k in gdat.indxpnts:
                            gdat.indxparapsfnxpos[k] = cntr
                            listminmpara.append(-1.)
                            listminmpara.append(1.)
                            listnamepara.append('xposdelt%04d' % k)
                            listlablpara.append([r'$\Delta x_{%d}$' % k, ''])
                            cntr += 1

                            gdat.indxparapsfnypos[k] = cntr
                            listminmpara.append(-1.)
                            listminmpara.append(1.)
                            listnamepara.append('yposdelt%04d' % k)
                            listlablpara.append([r'$\Delta y_{%d}$' % k, ''])
                            cntr += 1
                    
                    if gdat.typefittpsfncnts == 'vari':
                        gdat.indxparapsfnflux = np.empty(gdat.fitt.numbpnts, dtype=int)
                        for k in gdat.indxpnts:
                            gdat.indxparapsfnflux[k] = cntr
                            listminmpara.append(0.)
                            listmaxmpara.append(5. * np.amax(gdat.cntpdatasexp))
                            listnamepara.append('cnts%04d' % k)
                            listlablpara.append(['$C_{%d}$' % k, 'e$^-$/s'])
                            cntr += 1
                    
                    gdat.indxparapsfnback = np.empty(1, dtype=int)
                    gdat.indxparapsfnback[0] = cntr
                    listminmpara.append(0.)
                    listmaxmpara.append(np.amax(gdat.cntpdatasexp))
                    listnamepara.append('back')
                    listlablpara.append(['$B$', 'e$^-$/s/px'])
                    cntr += 1
                    
                    if gdat.fitt.typepsfnshap == 'gauscirc':
                        gdat.indxparapsfnpsfn = np.empty(1, dtype=int)
                        gdat.indxparapsfnpsfn[0] = cntr
                        listminmpara.append(0.5)
                        listmaxmpara.append(2.)
                        listnamepara.append('sigmpsfn')
                        listlablpara.append([r'$\sigma$', 'px'])
                        cntr += 1
                    
                    if gdat.fitt.typepsfnshap == 'gauselli':
                        gdat.indxparapsfnpsfn = np.empty(4, dtype=int)
                        gdat.indxparapsfnpsfn[0] = cntr
                        listminmpara.append(0.5)
                        listmaxmpara.append(2.)
                        listnamepara.append('sigmpsfnxpos')
                        listlablpara.append([r'$\sigma_x$', 'px'])
                        cntr += 1

                        gdat.indxparapsfnpsfn[1] = cntr
                        listminmpara.append(0.5)
                        listmaxmpara.append(2.)
                        listnamepara.append('sigmpsfnypos')
                        listlablpara.append([r'$\sigma_y$', 'px'])
                        cntr += 1

                        gdat.indxparapsfnpsfn[2] = cntr
                        listminmpara.append(-10.)
                        listmaxmpara.append(10.)
                        listnamepara.append('fracskewpsfnxpos')
                        listlablpara.append([r'$\chi_x$', ''])
                        cntr += 1

                        gdat.indxparapsfnpsfn[3] = cntr
                        listminmpara.append(-10.)
                        listmaxmpara.append(10.)
                        listnamepara.append('fracskewpsfnypos')
                        listlablpara.append([r'$\chi_y$', ''])
                    
                    if gdat.fitt.typepsfnshap == 'empi':
                        gdat.indxparapsfnpsfn = np.empty(gdat.numbparapsfnempi, dtype=int)
                        for r in gdat.indxparapsfnempi:
                            gdat.indxparapsfnpsfn[r] = cntr
                            listminmpara.append(0.)
                            listmaxmpara.append(np.amax(gdat.cntpdatasexp))
                            listnamepara.append('amplpsfnempi%04d' % r)
                            listlablpara.append(['$\rho_{%d}$' % r, ''])
                            cntr += 1
                    
                    gdat.numbparapsfn = cntr
                    gdat.indxparapsfn = np.arange(gdat.numbparapsfn)
                    listscalpara = ['self' for k in gdat.indxparapsfn]
                    listmeangauspara = None
                    liststdvgauspara = None
                    
                    listminmpara = np.array(listminmpara)
                    listmaxmpara = np.array(listmaxmpara)
                    
                    boolforcrepr = gdat.boolplotcntp

                    dictlablscalparaderi = None
                    if gdat.fitt.catl['xpos'].size > 1:
                        if dictlablscalparaderi is None:
                            dictlablscalparaderi = dict()
                        dictlablscalparaderi['raticont'] = [['$f_c$', ''], 'self']
                    if gdat.fitt.catl['xpos'][0] == (gdat.numbside[p] - 1.) / 2. and gdat.fitt.catl['ypos'][0] == (gdat.numbside[p] - 1.) / 2. \
                                                                                                            and gdat.fitt.catl['xpos'].size == 1:
                        if dictlablscalparaderi is None:
                            dictlablscalparaderi = dict()
                        dictlablscalparaderi['fraccent'] = [['$f_p$', ''], 'self']
                    retr_dictderi = retr_dictderipsfn

                    dictsamp = tdpy.samp(gdat, numbsampwalk, retr_llik, listnamepara, listlablpara, listscalpara, \
                                         listminmpara, listmaxmpara, numbsampburnwalkinit=numbsampburnwalkinit, \
                                                retr_dictderi=retr_dictderi, \
                                                dictlablscalparaderi=dictlablscalparaderi, \
                                                numbsampburnwalk=numbsampburnwalk, \
                                                boolforcrepr=boolforcrepr, \
                                                pathbase=gdat.pathvisutargsexp, strgextn=strgextn, \
                                                #typefileplot=typefileplot, \
                                                )
                    
                    if gdat.boolplotcntp:
                        
                        strgsavepsfn = gdat.strgsave + gdat.fitt.typepsfnshap
                        
                        # plot the posterior median image, point source models and the residual
                        for namevarb in ['cntpmodlpntssexp', 'cntpmodlsexp', 'cntpresisexp']:
                            setattr(gdat, namevarb, dictsamp[namevarb])
                            strgtitl = gdat.strgtitlcntpplot
                            boolresi = namevarb == 'cntpresisexp'
                            for typecntpscal in gdat.listtypecntpscal:
                                plot_cntp(gdat, np.median(dictsamp[namevarb], 0), p, o, typecntpscal, gdat.pathvisutargsexp, namevarb + gdat.strgsavepsfn,
                                                                                                                 'fitt', strgtitl=strgtitl, boolresi=boolresi)
                        for typecntpscal in gdat.listtypecntpscal:
                            plot_cntp(gdat, gdat.cntpdatasexp, p, o, typecntpscal, gdat.pathvisutargsexp, 'cntpdatasexp_' + gdat.strgsavepsfn, \
                                                                                                                  'fitt', strgtitl=gdat.strgtitlcntpplot)
                
                    dictpmed = dict()
                    for name in listnamepara:
                        dictpmed[name] = np.median(dictsamp[name])
                
                    print('dictpmed')
                    print(dictpmed)
                    print('Writing to %s...' % path)
                    pd.DataFrame.from_dict(dictpmed).to_csv(path)
                    #pd.DataFrame.from_dict(dictpmed).to_csv(path, index=False)
                
                else:
                    print('Reading posterior median PRF parameters from %s...' % path)
                    dictpmed = pd.read_csv(path).to_dict(orient='list')
                    print('dictpmed')
                    print(dictpmed)
                
                gdat.fitt.catl['cnts'] = np.empty(gdat.fitt.numbpnts[0])
                for k in gdat.indxpnts:
                    gdat.fitt.catl['cnts'][k] = dictpmed['cnts%04d' % k]
                if gdat.fitt.typepsfnshap == 'gauscirc':
                    gdat.fitt.parapsfn = np.empty(1)
                    gdat.fitt.parapsfn[0] = dictpmed['sigmpsfn']
                if gdat.fitt.typepsfnshap == 'gauselli':
                    gdat.fitt.parapsfn = np.empty(2)
                    gdat.fitt.parapsfn[0] = dictpmed['sigmpsfnxpos']
                    gdat.fitt.parapsfn[1] = dictpmed['sigmpsfnypos']
                    gdat.fitt.parapsfn[2] = dictpmed['fracskewpsfnxpos']
                    gdat.fitt.parapsfn[3] = dictpmed['fracskewpsfnypos']
                if gdat.fitt.typepsfnshap == 'empi':
                    gdat.fitt.parapsfn = np.empty(gdat.numbparapsfnempi)
                    for r in gdat.indxparapsfnempi:
                        gdat.fitt.parapsfn[r] = dictpmed['amplpsfnempi%04d' % r]
            
            else:
                
                if gdat.fitt.typepsfnshap == 'gauscirc':
                    gdat.fitt.parapsfn = np.empty(1)
                    gdat.fitt.parapsfn[0] = gdat.fitt.sigmpsfn
                elif gdat.fitt.typepsfnshap == 'gauselli':
                    gdat.fitt.parapsfn = np.empty(4)
                    gdat.fitt.parapsfn[0] = gdat.fitt.sigmpsfnxpos
                    gdat.fitt.parapsfn[1] = gdat.fitt.sigmpsfnypos
                    gdat.fitt.parapsfn[2] = gdat.fitt.fracskewpsfnxpos
                    gdat.fitt.parapsfn[3] = gdat.fitt.fracskewpsfnypos
                elif gdat.fitt.typepsfnshap == 'data':
                    pass
                else:
                    raise Exception('')
            
            if gdat.fitt.typepsfnshap != 'data':
                gdat.dictoutp['raticont'] = retr_raticonttotl(gdat, p, gdat.fitt.catl['xpos'], gdat.fitt.catl['ypos'], gdat.fitt.catl['cntsesti'], gdat.fitt.parapsfn)
            
            if gdat.boolanim:
                if gdat.boolanimframtotl:
                    numbplotanim = gdat.numbtime[p][o]
                else:
                    numbplotanim = 100
                # time indices to be included in the animation
                gdat.indxtimeanim = np.linspace(0., gdat.numbtime[p][o] - 1., numbplotanim).astype(int)
                # get time string
                if not gdat.boolsimutargsynt:
                    objttime = astropy.time.Time(gdat.listtime[p][o], format='jd', scale='utc')#, out_subfmt='date_hm')
                    listtimelabl = objttime.iso
                else:
                    listtimelabl = gdat.listtime[p][o].astype(str)
            
            # plot data with initial catalogs (after PM correction)
            if gdat.boolplotcntp:
                for typecntpscal in gdat.listtypecntpscal:
                    strgtitl = gdat.strgtitlcntpplot
                    plot_cntp(gdat, gdat.cntpdatasexp, p, o, typecntpscal, gdat.pathvisutargsexp, 'cntpdatasexp', 'refr', strgtitl=strgtitl)
            
            # plot a histogram of data counts
            if gdat.boolplothhistcntp:
                plot_histcntp(gdat, gdat.cntpdata, gdat.pathvisutarg, 'cntpdata', gdat.strgsave) 
            
            for e in gdat.indxanls:
                gdat.listnois[p][o][e] = np.zeros((gdat.numboffs, gdat.numboffs))
            for x in gdat.indxoffs:
                for y in gdat.indxoffs:
                    
                    for e in gdat.indxanls:
                        gdat.fitt.arrytser[p][o][e][x][y] = np.empty((gdat.numbtime[p][o], 3, gdat.fitt.numbpnts[e] + 1))
                        gdat.fitt.arrytser[p][o][e][x][y][:, 0, :] = gdat.listtime[p][o][:, None]
                        gdat.fitt.arrytserinit[p][o][e][x][y] = np.empty((gdat.numbtime[p][o], 3, gdat.fitt.numbpnts[e] + 1))
                        gdat.fitt.arrytserinit[p][o][e][x][y][:, 0, :] = gdat.listtime[p][o][:, None]
                    
                    if not gdat.boolfittoffs and (x != 1 or y != 1):
                        continue
                    
                    print('Offset indices: %d %d' % (x, y))
                    updt_strgsave(gdat, strgchun, x, y, p, o)
                    
                    for e, nameanls in enumerate(gdat.listnameanls):
                    
                        ## paths
                        pathsaverflx = gdat.pathdatatarg + 'tser%s_%s' % (gdat.strgnorm, nameanls) + gdat.strgsave + '.csv'
                        pathsaverflxtarg = gdat.pathdatatarg + 'tser%starg_%s' % (gdat.strgnorm, nameanls) + gdat.strgsave + '.csv'
                        pathsavemeta = gdat.pathdatatarg + 'metaregr_%s' % (nameanls) + gdat.strgsave + '.csv'
                        
                        # PSF photometry
                        if not os.path.exists(pathsaverflx):
                            
                            if nameanls == 'psfn':
                            
                                print('Performing PSF photometry...')
                                timeinit = timemodu.time()
                                
                                ## introduce the positional offset
                                xpostemp = np.copy(gdat.fitt.catl['xpos'])
                                ypostemp = np.copy(gdat.fitt.catl['ypos'])
                                xpostemp[0] = gdat.fitt.catl['xpos'][0] + gdat.listoffsposi[x]
                                ypostemp[0] = gdat.fitt.catl['ypos'][0] + gdat.listoffsposi[y]
                                
                                if gdat.quat is not None:
                                    # add temporal dimension
                                    cntstemp = np.ones((gdat.numbtime[p][o], gdat.fitt.numbpnts[e]))
                                    xpostemp = xpostemp[None, :] + gdat.quat[:, None, 1]
                                    ypostemp = ypostemp[None, :] + gdat.quat[:, None, 0]
                                
                                if gdat.booldiag:
                                    if gdat.cntpdata.shape[2] != gdat.numbtime[p][o]:
                                        print('')
                                        print('')
                                        print('')
                                        print('gdat.cntpdata')
                                        summgene(gdat.cntpdata)
                                        raise Exception('Last dimension of cntpdata should be same as gdat.numbtime[p][o].')
                                
                                gdat.cntpdataflat = gdat.cntpdata.reshape((-1, gdat.numbtime[p][o]))
                                gdat.variflat = gdat.vari.reshape((-1, gdat.numbtime[p][o]))
                                gdat.indxpixlnzer = np.where((gdat.cntpdataflat != 0).all(axis=1))[0]
                                
                                gdat.covafittcnts = np.empty((gdat.numbtime[p][o], gdat.fitt.numbpnts[e] + 1, gdat.fitt.numbpnts[e] + 1))
                                
                                if gdat.typeverb > 1:
                                    print('Solving for the best-fit raw light curves of the sources...')
                                
                                # solve the linear system
                                matrdesi = np.ones((gdat.indxpixlnzer.size, gdat.fitt.numbpnts[e] + 1))
                                
                                if gdat.booldiag:
                                    if gdat.fitt.numbpnts[e] != len(gdat.fitt.catl['xpos']):
                                        raise Exception('')

                                if gdat.indxpixlnzer.size != gdat.numbpixl:
                                    print('Some pixels are zero!')
                                    print('gdat.indxpixlnzer')
                                    print(gdat.indxpixlnzer)
                                    print('gdat.numbpixl')
                                    print(gdat.numbpixl)
                                    #raise Exception('')
                                
                                for t in gdat.indxtime[p][o]:
                                    
                                    if gdat.fitt.typepsfnshap == 'data':
                                        matrdesi[:, 0] = gdat.cntpdatapsfn.flatten()
                                    else:
                                        for k in np.arange(gdat.fitt.numbpnts[0]):
                                            if gdat.quat is None:
                                                cntpmodltemp = retr_cntpmodl(gdat, p, 'fitt', xpostemp[k, None], ypostemp[k, None], \
                                                                                                    np.array([1.]), 0., gdat.fitt.parapsfn, 'pnts')
                                            else:
                                                cntpmodltemp = retr_cntpmodl(gdat, p, 'fitt', xpostemp[t, k, None], ypostemp[t, k, None], cntstemp[t, k, None], 0., \
                                                                                                                     gdat.fitt.parapsfn, 'pnts')
                                            matrdesi[:, k] = cntpmodltemp.flatten()[gdat.indxpixlnzer]
                                    gdat.fitt.arrytserinit[p][o][e][x][y][t, 1, :], gdat.covafittcnts[t, :, :] = \
                                                               retr_mlikregr(gdat.cntpdataflat[gdat.indxpixlnzer, t], matrdesi, gdat.variflat[gdat.indxpixlnzer, t])
                                
                                for k in gdat.fitt.indxcomp[e]:
                                    gdat.fitt.arrytserinit[p][o][e][x][y][:, 2, k] = np.sqrt(gdat.covafittcnts[:, k, k])
                                    
                                    if not np.isfinite(gdat.fitt.arrytserinit[p][o][e][x][y][:, 2, k]).all():
                                        print('temp: error went NaN because of negative covariance. Reseting error to 1e-2')
                                        gdat.fitt.arrytserinit[p][o][e][x][y][:, 2, k] = 1e-2
                                        print('k')
                                        print(k)
                                        print('gdat.covafittcnts[:, k, k]')
                                        summgene(gdat.covafittcnts[:, k, k])
                                        #raise Exception('')
            
                                if False and gdat.booldetrcbvs:
                                    gdat.varifittcnts = gdat.stdvfittcnts[e]**2
                                    gdat.fitt.arrytser[p][o][e] = np.copy(gdat.fitt.arrytserinit[p][o][e])
                                    # subtract CBVs
                                    print('Solving for the detrended target light curve using the CBVs and the raw light curve...')
                                    gdat.mlikamplcbvs, gdat.covaamplcbvs = retr_mlikregr(gdat.fitt.arrytser[p][o][e][:, 0], gdat.cbvstmpt, gdat.varifittcnts[:, 0])
                                    rflxcbvs = gdat.mlikamplcbvs[None, :] * gdat.cbvstmpt
                                    rflxcbvstotl = np.sum(rflxcbvs[:, :-1], 1)
                                    gdat.fitt.arrytser[p][o][e][:, 0] -= rflxcbvstotl
                                else:
                                    gdat.fitt.arrytser[p][o][e][x][y] = gdat.fitt.arrytserinit[p][o][e][x][y]
                        
                                timefinl = timemodu.time()
                                print('Done in %g seconds.' % (timefinl - timeinit))
                                
                            if (x == 1 and y == 1) and nameanls == 'aper':
                                print('Performing aperture photometry...')
                                
                                print('Number of apertures: %d' % gdat.numbaper)
                                #gdat.listpixlapertargpair = [[] for c in gdat.indxaper]
                                #for p in gdat.indxaper:
                                #    for w in range(len(gdat.listpixlaperpair[c])):
                                #        gdat.listpixlapertargpair[c] = [gdat.listpixlapertarg[c][0][w], gdat.listpixlapertarg[c][1][w]]
                                #    for x in gdat.indxside:
                                #        for y in gdat.indxside:
                                #            if not (x, y) in (gdat.listpixlaperpair[c]:
                                #                gdat.listpixlaperback[c][0].append(x)
                                #                gdat.listpixlaperback[c][1].append(y)
                                #

                                timeinit = timemodu.time()
                                
                                gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0] = np.mean(gdat.cntpdata[gdat.listpixlapertarg[p][0], gdat.listpixlapertarg[p][1], :], axis=0)
                                gdat.fitt.arrytser[p][o][e][x][y][:, 1, 1] = np.mean(np.mean(gdat.cntpdata, axis=0), 0) - gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0]
                                
                                vari = np.zeros(gdat.numbtime[p][o]) + np.mean(gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0])
                                matrdesi = np.ones((gdat.numbtime[p][o], 2))
                                matrdesi[:, 0] = gdat.fitt.arrytser[p][o][e][x][y][:, 1, 1]
                                coef, cova = retr_mlikregr(gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0], matrdesi, vari)
                                gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0] -= coef[0] * gdat.fitt.arrytser[p][o][e][x][y][:, 1, 1]
                                
                                print('Aperture sums took %g seconds.' % (timemodu.time() - timeinit))

                            # filter based on background
                            #indxgoodbkgd = np.where(gdat.fitt.arrytser[p][o][e][x][y][:, 1, -1] < 3. * np.median(gdat.fitt.arrytser[p][o][e][x][y][:, 1, -1]))[0]
                            #gdat.fitt.arrytser[p][o][e][x][y] = gdat.fitt.arrytser[p][o][e][x][y][indxgoodbkgd, :, :]
                            
                            # indices where time series is not NaN
                            
                            indxtimegood = np.where(np.isfinite(gdat.fitt.arrytser[p][o][e][x][y][:, :, :]).all(axis=1).all(axis=1))[0]

                            gdat.medifittcnts = np.median(gdat.fitt.arrytser[p][o][e][x][y][indxtimegood, 1, :], 0)
                            print('Median flux of the light curve of the central source is %g ADU.' % gdat.medifittcnts[0])
                            
                            gdat.medifittcntsinit = np.median(gdat.fitt.arrytser[p][o][e][x][y][indxtimegood[:int(indxtimegood.size*0.1)], 1, :], 0)
                            print('Median flux of the initial 10%% of the light curve of the central source is %g ADU.' % gdat.medifittcntsinit[0])
                            
                            print('Normalizing the time-series flux...')
                            # normalize fluxes to get relative fluxes
                            if gdat.boolnorm:
                                if gdat.typenorm == 'medi':
                                    if gdat.booldiag:
                                        if not np.isfinite(gdat.medifittcnts).all():
                                            print('')
                                            print('')
                                            print('')
                                            raise Exception('not np.isfinite(gdat.medifittcnts).all()')

                                    gdat.fitt.arrytser[p][o][e][x][y][:, 1:3, :] /= gdat.medifittcnts[None, None, :]
                                
                                if gdat.typenorm == 'mediinit':
                                    if gdat.booldiag:
                                        if not np.isfinite(gdat.medifittcntsinit).all():
                                            print('')
                                            print('')
                                            print('')
                                            raise Exception('not np.isfinite(gdat.medifittcntsinit).all()')

                                    gdat.fitt.arrytser[p][o][e][x][y][:, 1:3, :] /= gdat.medifittcntsinit[None, None, :]
                                
                            gdat.fitt.tserfile[p][o][e][x][y] = np.empty((indxtimegood.size, 1+2*gdat.fitt.numbcomp[e]))
                            gdat.fitt.tserfile[p][o][e][x][y][:, 0] = gdat.fitt.arrytser[p][o][e][x][y][indxtimegood, 0, 0]
                            for k in gdat.fitt.indxcomp[e]:
                                gdat.fitt.tserfile[p][o][e][x][y][:, 2*k+1] = gdat.fitt.arrytser[p][o][e][x][y][indxtimegood, 1, k]
                                gdat.fitt.tserfile[p][o][e][x][y][:, 2*k+2] = gdat.fitt.arrytser[p][o][e][x][y][indxtimegood, 2, k]
                            
                            if gdat.booldiag:
                                if not np.isfinite(gdat.fitt.tserfile[p][o][e][x][y][indxtimegood, :3]).all():
                                    print('')
                                    print('')
                                    print('')
                                    raise Exception('not np.isfinite(gdat.fitt.tserfile[p][o][e][x][y][indxtimegood, :3]).all()')
                            
                                if not np.isfinite(gdat.fitt.tserfile[p][o][e][x][y][indxtimegood, :]).all():
                                    print('')
                                    print('')
                                    print('')
                                    raise Exception('not np.isfinite(gdat.fitt.tserfile[p][o][e][x][y][indxtimegood, :]).all()')
                            
                            # write the light curves to the disk
                            print('Writing all light curves to %s...' % pathsaverflx)
                            np.savetxt(pathsaverflx, gdat.fitt.tserfile[p][o][e][x][y], delimiter=',', header=gdat.strgheadtotl)
                            
                            # write the target light curve to the disk
                            print('Writing the target light curve to %s...' % pathsaverflxtarg)
                            np.savetxt(pathsaverflxtarg, gdat.fitt.tserfile[p][o][e][x][y][:, :3], delimiter=',', header=gdat.strgheadtarg)
                            
                            # write the meta data to the disk
                            print('Writing meta data to %s...' % pathsavemeta)
                            arry = np.empty((gdat.medifittcnts.size, 2), dtype=object)
                            arry[:, 0] = gdat.fitt.catl['labl']
                            arry[-1, 0] = 'Background'
                            arry[:, 1] = gdat.medifittcnts
                            np.savetxt(pathsavemeta, arry, delimiter=',', header='Target,Temporal median counts for each component', fmt="%s %.3g")
                                
                            if gdat.fitt.typepsfnshap != 'data':
                                print('Evaluating the model at all time bins...')
                                cntpbacktser = gdat.fitt.arrytser[p][o][e][x][y][:, 1, -1]
                                timeinit = timemodu.time()
                                gdat.cntpmodl = np.empty_like(gdat.cntpdata)
                                gdat.cntpmodlpnts = np.empty_like(gdat.cntpdata)
                                for t in gdat.indxtime[p][o]:
                                    if gdat.quat is not None:
                                        xpostemptemp = xpostemp[t, :]
                                        ypostemptemp = ypostemp[t, :]
                                    else:
                                        xpostemptemp = xpostemp
                                        ypostemptemp = ypostemp
                                        
                                    gdat.cntpmodlpnts[:, :, t] = retr_cntpmodl(gdat, p, 'fitt', xpostemptemp, ypostemptemp, \
                                                                                gdat.fitt.arrytser[p][o][e][x][y][t, 1, :-1], 0., gdat.fitt.parapsfn, 'pnts')
                                    gdat.cntpmodl[:, :, t] = retr_cntpmodl(gdat, p, 'fitt',xpostemptemp, ypostemptemp, gdat.fitt.arrytser[p][o][e][x][y][t, 1, :-1], \
                                                                                                                   cntpbacktser[t], gdat.fitt.parapsfn, 'pnts')
                                
                                timefinl = timemodu.time()
                                print('Done in %g seconds.' % (timefinl - timeinit))
                                    
                                gdat.cntpresi = gdat.cntpdata - gdat.cntpmodl
                                
                                chi2 = np.mean(gdat.cntpresi**2 / gdat.cntpdata) + 2 * gdat.fitt.numbpnts
                                
                                ## temporal medians
                                for strg in ['modl', 'modlpnts', 'resi']:
                                    cntp = getattr(gdat, 'cntp' + strg)
                                    setattr(gdat, 'cntp' + strg + 'tmed', np.nanmedian(cntp, axis=-1))
                            
                                # plot a histogram of data counts
                                if gdat.boolplothhistcntp:
                                    plot_histcntp(gdat, gdat.cntpresi, gdat.pathvisutarg, 'cntpresi')
                    
                                if gdat.boolplotcntp:
                                    for typeplotcntp in listtypeplotcntp:
                                        #for nameplotcntp in ['cntpmodl', 'cntpmodlpnts', 'cntpresi']:
                                        for nameplotcntp in []:
                                            
                                            # make animation plot
                                            if typeplotcntp == 'anim':
                                                pathanim = retr_pathvisu(gdat, gdat.pathvisutarg, nameplotcntp, gdat.strgsave, typevisu='anim')
                                                if os.path.exists(pathanim):
                                                    continue

                                                print('Making an animation of frame plots...')
                                                strg = ''
                                            else:
                                                strg = 'tmed'
                                            
                                            cntptemp = getattr(gdat, nameplotcntp + strg)
                                            
                                            boolresi = nameplotcntp.startswith('cntpresi')
                                            
                                            for typecntpscal in gdat.listtypecntpscal:
                                            
                                                if typeplotcntp == 'tmed':
                                                    if nameplotcntp == 'cntpdata':
                                                        listtypecatlplot = gdat.listtypecatlplot
                                                    else:
                                                        listtypecatlplot = ['fitt']
                                                    for typecatlplot in listtypecatlplot:
                                                        strgtitl = gdat.strgtitlcntpplot
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('typecatlplot')
                                                        print(typecatlplot)
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        print('')
                                                        plot_cntp(gdat, cntptemp, p, o, typecntpscal, gdat.pathvisutarg, \
                                                                                                        nameplotcntp + strg, gdat.strgsave, typecatlplot, \
                                                                                                                                   xposoffs=gdat.listoffsposi[x], \
                                                                                                                                   yposoffs=gdat.listoffsposi[y], \
                                                                                                                                   strgtitl=strgtitl, boolresi=boolresi)
                                                if x == 1 and y == 1 and typeplotcntp == 'anim' and nameplotcntp == 'cntpresi':
                                                    # color scales
                                                    setp_cntp(gdat, nameplotcntp, typecntpscal)
                                                    
                                                    vmin = getattr(gdat, 'vmin' + nameplotcntp + typecntpscal)
                                                    vmax = getattr(gdat, 'vmax' + nameplotcntp + typecntpscal)
                                            
                                                    args = [gdat, cntptemp, p, o, typecntpscal, nameplotcntp]
                                                    kwag = { \
                                                            'boolresi': boolresi, \
                                                            #'listindxpixlcolr': gdat.listpixlaper, \
                                                            'listtimelabl':listtimelabl, \
                                                            'vmin':vmin, 'vmax':vmax, \
                                                            'tser':gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0], \
                                                            'time':gdat.listtime[p][o], \
                                                           }
                                                    listpath = anim_cntp(*args, **kwag)
                                            
                                                    # make animation
                                                    cmnd = 'convert -delay 20 '
                                                    for path in listpath:
                                                        cmnd += '%s ' % path
                                                    cmnd += '%s' % pathanim
                                                    os.system(cmnd)
                                                    
                                                    # delete images
                                                    for path in listpath:
                                                        os.system('rm %s' % path)
         
                    
                        else:
                            print('Skipping the regression...')
                        
                            #print('Reading from %s...' % pathsavemeta)
                            #gdat.medifittcnts = np.loadtxt(pathsavemeta, delimiter=',', skiprows=1)
                            
                            print('Reading from %s...' % pathsaverflx)
                            gdat.fitt.tserfile[p][o][e][x][y] = np.loadtxt(pathsaverflx, delimiter=',', skiprows=1)
                            gdat.listtime[p][o] = gdat.fitt.tserfile[p][o][e][x][y][:, 0]
                            
                            for k in gdat.fitt.indxcomp[e]:
                                gdat.fitt.arrytser[p][o][e][x][y][:, 0, k] = gdat.fitt.tserfile[p][o][e][x][y][:, 0]
                                gdat.fitt.arrytser[p][o][e][x][y][:, 1:3, k] = gdat.fitt.tserfile[p][o][e][x][y][:, 1+2*k:1+2*(k+1)]

                    if gdat.booldiag:
                        for a in range(gdat.listtime[p][o].size):
                            if a != gdat.listtime[p][o].size - 1 and gdat.listtime[p][o][a] >= gdat.listtime[p][o][a+1]:
                                print('gdat.listtime[p][o]')
                                summgene(gdat.listtime[p][o])
                                raise Exception('')
                    
                    # assess noise in the light curve
                    for e in gdat.indxanls:
                        if x == 1 and y == 1:
                            arryrflxrbn = np.copy(gdat.fitt.arrytser[p][o][e][x][y][:, :, 0])
                            arryrflxrbn[:, 1] = gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0] - \
                                                            scipy.ndimage.median_filter(gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0], size=50)
                            arryrflxrbn = miletos.rebn_tser(arryrflxrbn, delt=1. / 24.)
                            gdat.listnois[p][o][e][x, y] = 1e6 * np.nanstd(arryrflxrbn[:, 1]) / np.median(gdat.fitt.arrytser[p][o][e][x][y][:, 1, 0])
                    
                        if gdat.booldiag:
                            if len(gdat.fitt.arrytser[p][o][e][x][y]) == 0:
                                raise Exception('')
                        
                        # construct the LQF
                        if x == 1 and y == 1:
                            rflxbkgd = gdat.fitt.arrytser[p][o][e][1][1][:, 1, -1]
                            gdat.listindxtimegoodlyqf[p][o][e] = retr_indxtimegoodlyqf(gdat, rflxbkgd)

                        if gdat.boolplotrflx:
                            print('Plotting light curves...')
                            
                            #if gdat.booldetrcbvs:
                            #    plot_lcur(gdat, gdat.fitt.arrytser[p][o][:, 1, 0, x, y], gdat.fitt.arrytser[p][o][:, 2, 0, x, y], \
                            #                                                                    0, p, o, '_' + strgchun, 'detrcbvs')
                                
                            # plot the light curve of the sources and background
                            for k in gdat.fitt.indxcomp[e]:
                                if x == 1 and y == 1 or k == 0:
                                    
                                    # grab the CDPP
                                    if k == 0:
                                        cdpp = gdat.listnois[p][o][0][x, y]
                                    else:
                                        cdpp = None
                                    
                                    plot_lcur(gdat, gdat.fitt.arrytser[p][o][e][x][y][:, 1, k], gdat.fitt.arrytser[p][o][e][x][y][:, 2, k], k, p, o, '_' + strgchun, \
                                                                                               gdat.listnameanls[e], numbcomp=gdat.fitt.numbcomp[e], cdpp=cdpp)
                                    
                                    if gdat.boolrefrtser[p][o] and x == 1 and y == 1 and k == 0:
                                        # plot light curve without the lygos quality flag (LQF)
                                        plot_lcurcomp(gdat, gdat.fitt.arrytser[p][o][e][x][y][:, 1, k], \
                                                            gdat.fitt.arrytser[p][o][e][x][y][:, 2, k], \
                                                            k, p, o, '_' + strgchun, gdat.listnameanls[e], numbcomp=gdat.fitt.numbcomp[e])
                                    
                                        # plot light curve with the LQF
                                        for rr in gdat.indxlyqf:
                                            plot_lcurcomp(gdat, gdat.fitt.arrytser[p][o][e][x][y][gdat.listindxtimegoodlyqf[p][o][e][rr], 1, k], \
                                                                gdat.fitt.arrytser[p][o][e][x][y][gdat.listindxtimegoodlyqf[p][o][e][rr], 2, k], \
                                                                k, p, o, '_' + strgchun, gdat.listnameanls[e], numbcomp=gdat.fitt.numbcomp[e])
                                    
                                    # plot light curves by zooming into the provided time interval
                                    if gdat.listlimttimetzom is not None:
                                        for p in range(len(gdat.listlimttimetzom)):
                                            plot_lcur(gdat, gdat.fitt.arrytser[p][o][e][x][y][:, 1, k], \
                                                                                gdat.fitt.arrytser[p][o][e][x][y][:, 2, k], k, p, o, '_' + strgchun, \
                                                                   'zoom', numbcomp=gdat.fitt.numbcomp[e], indxtimelimt=gdat.indxtimelimt[p], indxtzom=p)
                                    
                            # plot light curves of all (of the first quality cut) sources together
                            nameplot = 'tser%ssour' % gdat.strgnorm
                            path = retr_pathvisu(gdat, gdat.pathvisutarg, nameplot)
                            if not os.path.exists(path):
                                figr, axis = plt.subplots(gdat.fitt.numbcomp[e], 1, \
                                                                figsize=(2 * gdat.sizefigrside, 0.6 * gdat.fitt.numbcomp[e] * gdat.sizefigrside), sharex=True)
                                for k in gdat.fitt.indxcomp[e]:
                                    axis[k].plot(gdat.listtime[p][o][gdat.listindxtimegoodlyqf[p][o][e][0]] - gdat.timeoffs, \
                                                 gdat.fitt.arrytser[p][o][e][x][y][gdat.listindxtimegoodlyqf[p][o][e][0], 1, k], color='gray', \
                                                                                                                                        ls='', marker='.', ms=1)
                                    axis[k].set_title(gdat.fitt.lablcomp[k])
                                axis[int(gdat.fitt.numbcomp[e]/2)].set_ylabel('Relative flux')
                                axis[int(gdat.fitt.numbcomp[e]-1)].set_xlabel(gdat.labltime)
                                print('Writing to %s...' % path)
                                plt.savefig(path)
                                plt.close()

                                    
            for e in gdat.indxanls:
                nameanls = gdat.listnameanls[e]
                gdat.dictoutp['nois%s%s%s' % (gdat.liststrginst[p], strgchun, nameanls)] = gdat.listnois[p][o][e][1, 1]
                gdat.dictoutp['arryrflx'][nameanls][p][o] = gdat.fitt.arrytser[p][o][e][1][1][:, :, 0]
        
        gdat.dictoutp['listipnt'][p] = gdat.listipnt[p]
    if gdat.booldiag:
        for p in gdat.indxinst:
            for e in gdat.indxanls:
                nameanls = gdat.listnameanls[e]
                if len(gdat.dictoutp['arryrflx'][nameanls][p]) != len(gdat.dictoutp['listipnt'][p]):
                    print('')
                    print('')
                    print('')
                    print('len(gdat.dictoutp[arryrflx][nameanls][p])')
                    print(len(gdat.dictoutp['arryrflx'][nameanls][p]))
                    print('gdat.dictoutp[listipnt]')
                    summgene(gdat.dictoutp['listipnt'])
                    raise Exception('len(gdat.dictoutp[arryrflx][nameanls][p])')
    
    for name, valu in gdat.true.__dict__.items():
        gdat.dictoutp['true'+name] = valu
    
    for name, valu in gdat.fitt.__dict__.items():
        gdat.dictoutp['fitt'+name] = valu
    
    for name, valu in gdat.__dict__.items():
        if name.startswith('strgtitl'):
            continue
        if name.startswith('strghead'):
            continue
        gdat.dictoutp[name] = valu
    
    timefinltotl = timemodu.time()
    print('lygos ran in %g seconds.' % (timefinltotl - timeinittotl))
    print('')                
    
    return gdat.dictoutp


