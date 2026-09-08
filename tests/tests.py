import os, sys
import numpy as np

import pandas as pd
import lygos
import ephesos
import miletos
import tdpy
from tdpy import summgene

import matplotlib.pyplot as plt

# first, make sure that the environment variable $LYGOS_DATA_PATH is set to the folder, where you would like the output plots and data to appear

def cnfg_lsst_exop():
    '''
    Pipeline to simulate LSST
    '''   
    for typedata in ['simutoyy', 'simuskyy']:
        lygos.init( \
                   strgmast='TOI-1233', \
                   liststrginst=['LSST-r'], \
                   liststrgtypedata=liststrgtypedata, \
                  )


def cnfg_ULTRASAT():
    '''
    Pipeline to simulate TESSGeo light curves'''   
    
    strgmast = 'Vega'
    
    #nametarg = 'Earth'
    #nametarg = None
    
    listtime = [[np.array([0.])]]
    
    liststrgtypedata = ['simutargpartsynt', 'obsd']
    lygos.init( \
               strgmast=strgmast, \
                
               listtime=listtime, \
               #strgmast=strgmast, \
               #nametarg=nametarg, \
               maxmtmagcatl=6., \
               #numbside=[500, 50, 5, 5], \
               
               liststrginst=['ULTRASAT', 'TESS'], \
               liststrgtypedata=liststrgtypedata, \
              )


def cnfg_TESSGEO():
    '''
    Pipeline to simulate TESSGeo light curves'''   
    
    strgmast = 'TOI-1233'
    
    #nametarg = 'Earth'
    #nametarg = None
    
    liststrgtypedata = ['simutargpartsynt', 'simutargpartsynt', 'simutargpartsynt', 'obsd']
    for typedata in ['simuskyy']:
        lygos.init( \
                   strgmast=strgmast, \
                   #strgmast=strgmast, \
                   #nametarg=nametarg, \
                   #numbside=numbside, \
                   #[500, 50, 5, 5], \
                   
                   liststrginst=['TGEO-UV', 'TGEO-VIS', 'TGEO-IR', 'TESS'], \
                   liststrgtypedata=liststrgtypedata, \
                  )


def IRAS090263817():
   
    labltarg = 'IRAS09026-3817'
    rasctarg = 136.1388872
    decltarg = -38.4895431

    for k in range(1, 4):
        lygos.init(rasctarg=rasctarg, decltarg=decltarg, labltarg=labltarg, maxmnumbstar=k)


def ASASNN():
    
    # a string that will be used to query the target on MAST, which should be resolvable on MAST
    liststrgmast = [ \
                    'ASASSN-19bt', \
                    #'ASASNN-19fp', \
                   ]
    for strgmast in liststrgmast:
        lygos.init( \
                   strgmast=strgmast, \
                   strgclus='ASASSN', \
                  )


def tutorial():
    
    lygos.init(strgmast='WASP-121')
    lygos.init(toiitarg=1233)
    lygos.init(rasctarg=124.343, decltarg=decltarg)


def cnfg_TOI2406():

    lygos.init( \
               toiitarg=2406, \
              )


def cnfg_WASP121():
    
    lygos.init( \
               strgmast='WASP-121', \
               typepsfninfe='fixd', \
               listtsecsele=[7, 33, 34], \
              )
        

def cnfg_Sirius():
    
    lygos.init( \
         strgmast='Sirius', \
         liststrginst=['TESS'], \
         numbside=[51], \
         maxmtmagcatl=9., \
        )
        

def cnfg_TOI1233():
    
    lygos.init( \
         toiitarg=1233, \
         numbside=[21], \
        )
        
    lygos.init( \
         toiitarg=1233, \
         numbside=[51], \
        )
        

def cnfg_requests():
    
    lygos.init( \
         #strgmast='V563 Lyr', \
         
         # from Ozgur Basturk
         strgmast='HAT-P-19', \

         # from Ben Rackham
         #strgmast='Ross 619', \
         
         # from Andrew Vanderburg, white dwarf
         ticitarg=902906874, \
        
         boolfasttpxf=True, \
         numbside=[7], \
         #boolfittoffs=True, \
        )


def cnfg_Pleides():
    
    lygos.init( \
         strgmast='Electra', \
        )


def cnfg_KOI1003():
    
    lygos.init( \
         #ticitarg=122374527, \
         labltarg='KOI 1003', \
         strgmast='KOI-1003', \
         epocpmot=2019.3, \
        )


def cnfg_Luhman16():
   
    pmratarg = -2762.16 # [mas/yr]
    pmdetarg = 357.79 # [mas/yr]
    
    # exofop (J2015.5)
    #rasctarg = 162.328259
    #decltarg = -53.319372°
    
    # TICv8 (J2000)
    rasctarg = 162.328814
    decltarg = -53.319466
    
    lygos.init( \
         rasctarg=rasctarg, \
         decltarg=decltarg, \
         labltarg='Luhman 16', \
         booladdddiscdetr=True, \
         maxmnumbstar=5, \
         epocpmot=2019., \
         pmratarg=pmratarg, \
         pmdetarg=pmdetarg, \
         psfntype='lion', \
        )
        

def cnfg_movingobject():
    
    numbtargneig = 10
    
    dicttrue = dict()
    
    numbtarg = 1
    
    typepsfninfe = 'fixd'
        

    for k in range(1):
        strgtarg = 'ISOB%04d' % k
        labltarg = 'Synthetic ISO %d' % k
        
        dicttrue['xpostarg'] = 5. + np.random.rand()
        dicttrue['ypostarg'] = 5. + np.random.rand()
        dicttrue['boolssob'] = True
        dicttrue['anglvelotarg'] = np.random.rand() * 2. * np.pi
        dicttrue['disttarg'] = np.random.rand() * 400.
        dicttrue['tmagtarg'] = 10.
        dicttrue['pmxatarg'] = 1.
        dicttrue['pmyatarg'] = 1.
        dicttrue['xposneig'] = 5. + np.random.rand(numbtargneig)
        dicttrue['yposneig'] = 5. + np.random.rand(numbtargneig)
        dicttrue['tmagneig'] = 5. + np.random.rand(numbtargneig)
        
        lygos.init( \
             strgtarg=strgtarg, \
             labltarg=labltarg, \
             typedata='simutoyy', \
             dicttrue=dicttrue, \
                       
             strgclus='isob', \
             
             typepsfninfe=typepsfninfe, \

            )


def cnfg_syst(typeanls):
    '''
    Investigate systematics
    typemult:
        'sing': single source
        'doub': two sources
    '''
    # get the features of highly-contaminated sources in the TIC
    dictpopl = ephesos.retr_dictpopltic8('tici_prms_allm_hcon')
    
    if typeanls == 'obsd':
        typemult = None
        typedata = typeanls
    else:
        typedata, typemult = typeanls.split('_')
    
    pathbase = os.environ['LYGOS_DATA_PATH'] + '/syst/'
    pathimag = pathbase + 'imag/'
    
    numbside = 11

    if typemult == 'sing':
        numbsour = 2
        boolanim = False
        boolfittoffs = True
    else:
        numbsour = 100
        boolanim = False
        boolfittoffs = False
    
    indxsour = np.arange(numbsour)
    listtmag = np.empty(numbsour)
    listnois = np.empty(numbsour)
    listsepa = np.empty(numbsour)

    strgclus = 'systematics_' + typeanls

    boolplot = True

    dictfitt = dict()
    dicttrue = dict()
    for k in indxsour:
        
        #if typemult == 'sing':
        #    boolplot = True
        #else:
        #    boolplot = False
        
        if typemult == 'sing':
            typepsfninfe = 'locl'
        else:
            typepsfninfe = 'fixd'
            
        if typemult == 'sing' and k == 0:
            dicttrue['typepsfnshap'] = 'gauselli'
            dicttrue['sigmpsfnxpos'] = 0.9
            dicttrue['sigmpsfnypos'] = 1.2
            dicttrue['fracskewpsfnxpos'] = 0.4
            dicttrue['fracskewpsfnypos'] = 0.7
        
        if typemult == 'sing' and k == 1:
            dicttrue['typepsfnshap'] = 'gauscirc'
            
        if typedata == 'simutoyy' or typedata == 'inje':
            ticitarg = None
            if typemult == 'sing' or typemult == 'doub':
                dicttrue['tmagtarg'] = 10.
            else:
                dicttrue['tmagtarg'] = tdpy.icdf_self(np.random.rand(), 7., 20.)
            listtmag[k] = dicttrue['tmagtarg']
        
        if typedata == 'simutoyy':
            if typemult == 'sing':
                strgtarg = 'simugene%s%starg' % (typemult, dicttrue['typepsfnshap'])
            else:
                strgtarg = 'simugene%starg%04d' % (typemult, k)
            labltarg = 'Sim. Image, Sim. T=%.1f Source' % listtmag[k]
            print('labltarg')
            print(labltarg)
            if typemult == 'isol':
                dicttrue['cntpbackscal'] = 100. # [e-/s]
                dictfitt['cntpbackscal'] = 90. # [e-/s]
            
            if typemult == 'bkgd':
                dicttrue['cntpbackscal'] = 100. # [e-/s]
                dictfitt['cntpbackscal'] = 90. # [e-/s]
            
            if typemult == 'psfn':
                dicttrue['sigmpsfn'] = 1. # [px]
                dictfitt['sigmpsfn'] = 0.9 # [px]
            
            rasctarg = None
            decltarg = None
            if typemult == 'doub':
                cent = (numbside - 1.) / 2.
                offs = tdpy.icdf_self(np.random.rand(), 0., 1.)
                listsepa[k] = 2. * offs
                dicttrue['numbneig'] = 1
                dicttrue['xpostarg'] = cent - offs
                dicttrue['xposneig'] = np.array([cent + offs])
                dicttrue['yposneig'] = np.array([cent])
                dicttrue['tmagneig'] = np.array([10.])
            if typemult == 'blen':
                dicttrue['cntpbackscal'] = tdpy.icdf_self(np.random.rand(), 50., 300.) # [e-/s]
                dicttrue['numbneig'] = 10
                dicttrue['xposneig'] = tdpy.icdf_self(np.random.rand(dicttrue['numbneig']), 0., numbside - 1.)
                dicttrue['yposneig'] = tdpy.icdf_self(np.random.rand(dicttrue['numbneig']), 0., numbside - 1.)
                dicttrue['tmagneig'] = tdpy.icdf_self(np.random.rand(dicttrue['numbneig']), 10., 20.)

        if typedata == 'inje':
            strgtarg = 'injetarg%04d' % k
            labltarg = 'Real Image, Injected T=%.3g Source' % listtmag[k]
            rasctarg = np.random.rand() * 360.
            decltarg = -90. + np.random.rand() * 180.
        
        if typedata == 'obsd':
            strgtarg = None
            ticitarg = dictpopl['tici'][k]
            listtmag[k] = dictpopl['tmag'][k]
            dicttrue = None
            rasctarg = None
            decltarg = None
            labltarg = None

        dictoutp = lygos.init( \
                   strgclus=strgclus, \
                   
                   ticitarg=ticitarg, \
                   rasctarg=rasctarg, \
                   decltarg=decltarg, \
                   strgtarg=strgtarg, \
                   labltarg=labltarg, \
                   
                   boolanim=boolanim, \
                   boolplot=boolplot, \
                   boolfittoffs=boolfittoffs, \
        
                   typepsfninfe=typepsfninfe, \
                   
                   dictfitt=dictfitt, \
                   dicttrue=dicttrue, \
                   
                   boolmerg=False, \

                   seedrand=k, \

                   liststrgtypedata=liststrgtypedata, \
                  )
        
        if len(dictoutp['listnois']) > 0:
            listnois[k] = dictoutp['listnois'][0][0][1, 1]
    
    if typemult == 'doub':
        figr, axis = plt.subplots()
        axis.scatter(listsepa, listnois, s=2)
        axis.set_yscale('log')
        axis.set_ylabel('1-hour CDPP [ppm]')
        axis.set_xlabel('Separation [px]')
        #plt.tight_layout()
        path = pathimag + 'noissepa_%s.pdf' % typemult
        plt.savefig(path)
        plt.close()
    
    if typemult != 'sing' and typemult != 'doub':
        figr, axis = plt.subplots()
        axis.scatter(listtmag, listnois, s=2)
        axis.set_yscale('log')
        axis.set_ylabel('1-hour CDPP [ppm]')
        axis.set_xlabel('TESS Mag')
        #plt.tight_layout()
        path = pathimag + 'noistmag_%s.pdf' % typemult
        plt.savefig(path)
        plt.close()

    
def cnfg_GJ299():
    
    lygos.init( \
         #boolfittoffs=True, \
         labltarg='GJ 299', \
         ticitarg=334415465, \
         epocpmot=2019.3, \
        )
        

def cnfg_spec():
    
    path = os.environ['LYGOS_DATA_PATH'] + '/data/List_for_MIT_pilot.txt'
    data = np.loadtxt(path, delimiter='\t', skiprows=1)
    numbtarg = data.shape[0]
    indxtarg = np.arange(numbtarg)
    for k in indxtarg:
        ticitarg = int(data[k, 2])
        lygos.init( \
             ticitarg=ticitarg, \
             labltarg='TIC %s' % ticitarg, \
             strgtarg='speculus_%s' % ticitarg, \
            )
        
    
def cnfg_saul():
    
    path = os.environ['LYGOS_DATA_PATH'] + '/data/list_saul.txt'
    strgclus = 'saul'

    pathbase = os.environ['LYGOS_DATA_PATH'] + '/%s/' % strgbase
    os.system('mkdir -p %s' % pathbase)

    listticitarg = []
    listlabl = []
    listrasc = []
    listdecl = []
    for line in open(pathfile):
        listline = line.split('  ')
        labltarg = listline[0]
        strgtarg = labltarg
        rasctarg = float(listline[1])
        decltarg = float(listline[2])
        intgresu = init( \
                        strgtarg=strgtarg, \
                        rasctarg=rasctarg, \
                        decltarg=decltarg, \
                        labltarg=labltarg, \
                        strgbase=strgbase, \
                        **kwag \
                       )


def cnfg_KeplerEBs():
    # Amaury
    strgclus = 'KeplerEBs'
    cnfg_arry(strgclus) 


def cnfg_SPECULOOS():
    
    strgclus = 'SPECULOOS'
    cnfg_arry(strgclus) 


def cnfg_arry(strgclus):
    
    pathdata = os.environ['LYGOS_DATA_PATH'] + '/data/'
    if strgclus == 'KeplerEBs':
        path = pathdata + 'KeplerEBs/Kepler_binaries_priority.csv'
    else:
        path = pathdata + 'SPECULOOS/final_catalog.csv'
    
    cntr = 0
    indx
    listtici = []
    print('Reading from %s...' % path)
    for line in open(path, 'r'):
        if cntr == 0:
            cntr += 1
            continue
        linesplt = line.split(',')
        if strgclus == 'KeplerEBs':
            listtici.append(int(linesplt[1]))
        else:
            listtici.append(int(linesplt[23]))
    
    numbtarg = len(listtici)
    indxtarg = np.arange(numbtarg)
    listintgresu = np.empty(numbtarg)
    for k in indxtarg:
        temp = lygos.init( \
                              ticitarg=listtici[k], \
                              strgclus=strgclus, \
                             )
        

def chec_runs():
    
    strg = 'spec1313'

    path = os.environ['LYGOS_DATA_PATH'] + '/'
    liststrgfile = fnmatch.filter(os.listdir(path), '%s_*' % strg)
    numb = len(liststrgfile)
    listbool = np.zeros(numb, dtype=bool)
    for k, strgfile in enumerate(liststrgfile):
        liststrgextn = fnmatch.filter(os.listdir(path + strgfile + '/data/'), 'rflx_*')
        if len(liststrgextn) == 1:
            listbool[k] = True


def cnfg_test347543557():

    ticitarg = 347543557
    labltarg = 'TIC 347543557'
    strgtarg = 'test347543557'
    listlimttimeplot = [[2458428, 2458430]]
    lygos.init( \
         ticitarg=ticitarg, \
         labltarg=labltarg, \
         strgtarg=strgtarg, \
         listlimttimeplot=listlimttimeplot, \
        )


def cnfg_GRB191016A():
    
    rasctarg = 30.2695
    decltarg = 24.5099
    labltarg = 'GRB191016A'
    listlimttimeplot = []
    for timedelt in [1.]:
        listlimttimeplot.append(2458772.67 + np.array([-timedelt, timedelt]))
    listtimeplotline = [2458772.67291666667]
    boolfittoffs = False#True
    boolcuttqual = False
    lygos.init( \
         rasctarg=rasctarg, \
         decltarg=decltarg, \
         labltarg=labltarg, \
         boolcuttqual=boolcuttqual, \
         boolfittoffs=boolfittoffs, \
         numbside=[9], \
         listtimeplotline=listtimeplotline, \
         listlimttimeplot=listlimttimeplot, \
        )


def cnfg_lindsey():
    
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    pathbase = os.environ['PERGAMON_DATA_PATH'] + '/featsupntess/'
    pathdata = pathbase + 'data/'
    pathimag = pathbase + 'imag/'
    os.system('mkdir -p %s' % pathdata)
    os.system('mkdir -p %s' % pathimag)
    for strgclus in [ \
                     'Cycle1-matched', \
                     'Cycle2-matched', \
                     'Cycle3-matched', \
                    ]:

        pathcsvv = pathdata + '%s.csv' % strgclus
        print('Reading from %s...' % pathcsvv)
        objtfile = open(pathcsvv, 'r')
        k = 0
        for line in objtfile:
            if k == 0:
                k += 1
                continue
            linesplt = line.split(',')
            labltarg = linesplt[2]
            c = SkyCoord('%s %s' % (linesplt[3], linesplt[4]), unit=(u.hourangle, u.deg))
            rasctarg = c.ra.degree
            decltarg = c.dec.degree
            
            print('labltarg')
            print(labltarg)
            print('linesplt[1]')
            print(linesplt[1])
            print('linesplt[2]')
            print(linesplt[2])
            print('rasctarg')
            print(rasctarg)
            print('decltarg')
            print(decltarg)
            
            dictoutp = lygos.init( \
                                  rasctarg=rasctarg, \
                                  decltarg=decltarg, \
                                  labltarg=labltarg, \
                                  #numbside=[5], \
                                  booldetrcbvs=False, \
                                  strgclus=strgclus, \
                                  booltpxflygo=False, \
                                 )
            
            listtsec = dictoutp['listtsec']
            numbtsec = len(listtsec)
            indxtsec = np.arange(numbtsec)
            for o in indxtsec:
                cmnd = 'cp %s%s/%s/imag/* %s%s/imag/' % (pathbase, strgclus, dictoutp['strgtarg'], pathbase, strgclus)
                print(cmnd)
                os.system(cmnd)
                pathsaverflxtarg = dictoutp['pathsaverflxtargsc%02d' % listtsec[o]]
                cmnd = 'cp %s %s%s/data/' % (pathsaverflxtarg, pathbase, strgclus)
                print(cmnd)
                os.system(cmnd)
            k += 1


def cnfg_DJ():
    
    listname = ['quasar_transients_2', 'quasar_gaia_locs', 'galRaDec', 'raDecs_transient']
    numbname = len(listname)
    indxname = np.arange(numbname)
    for n in indxname:
        strgclus = listname[n]
        path = os.environ['LYGOS_DATA_PATH'] + '/data/djjj/' + listname[n]
        print('Reading from %s...' % path)
        arry = np.loadtxt(path, delimiter=' ')
        rasc = arry[:, 0]
        decl = arry[:, 1]
        numbtarg = rasc.size
        indxtarg = np.arange(numbtarg)
        for k in indxtarg:
            labltarg = 'Target %04d' % k
            strgtarg = 'lygos_DJ_target_%04d' % k
            rasctarg = rasc[k]
            decltarg = decl[k]
            lygos.init( \
                 rasctarg=rasctarg, \
                 decltarg=decltarg, \
                 labltarg=labltarg, \
                 strgclus=strgclus, \
                 maxmnumbstar=1, \
                 strgtarg=strgtarg, \
                )
            #break
            #k += 1



def cnfg_TIC284856863():
    '''
    12 July 2021, young star with accretion disk from Max (TYC 2597-735-1)
    '''

    lygos.init( \
               boolregrforc=True, \
               maxmnumbstar=999, \
               maxmdmag=6, \
               ticitarg=284856863, \
               labltarg='TYC 2597-735-1', \
              )


def cnfg_ASASSN20qc():
    '''
    13 July 2021, AGN from DJ
    '''
    
    rasctarg = 63.260208 
    decltarg = -53.0727

    labltarg = 'ASASSN-20qc'
    
    refrlistlabltser = [['Michael']]
    path = os.environ['LYGOS_DATA_PATH'] + '/data/lc_2020adgm_cleaned_ASASSN20qc'
    print('Reading from %s...' % path)
    objtfile = open(path, 'r')
    k = 0
    linevalu = []
    for line in objtfile:
        if k == 0:
            k += 1
            continue
        linesplt = line.split(' ')
        linevalu.append([])
        for linesplttemp in linesplt:
            if linesplttemp != '':
                linevalu[k-1].append(float(linesplttemp))
        linevalu[k-1] = np.array(linevalu[k-1])
        k += 1
    linevalu = np.vstack(linevalu)
    refrarrytser = np.empty((linevalu.shape[0], 3))
    refrarrytser[:, 0] = linevalu[:, 0]
    refrarrytser[:, 1] = linevalu[:, 2]
    refrarrytser[:, 2] = linevalu[:, 3]
   
    dictmileinpt = dict()
    dictmileinpt['listtypemodl'] = ['supn']
    
    listnumbside = [[7], [11], [15]]
    #dictmileinpt['listlimttimemask'] = [[[[-np.inf, 2457000 + 2175], [2457000 + 2186.5, 2457000 + 2187.5]]]]
    dictmileinpt['listlimttimemask'] = [[[[2457000 + 2186.5, 2457000 + 2187.5]]]]
    for numbside in listnumbside:
        if numbside == 11:
            dictmileinpt['listtimescalbdtrspln'] = [0., 0.1, 0.5]
            boolfittoffs = True
        else:
            dictmileinpt['listtimescalbdtrspln'] = [0.]
            boolfittoffs = False

        dictoutp = lygos.init( \
                      boolplotrflx=True, \
                      boolplotcntp=True, \
                      boolfittoffs=boolfittoffs, \
                
                      refrlistlabltser=refrlistlabltser, \
                      refrarrytser=refrarrytser, \

                      labltarg=labltarg, \
                      
                      listtsecsele=[32], \
                      dictmileinpt=dictmileinpt, \
                      
                      timeoffs=2459000, \

                      numbside=numbside, \

                      rasctarg=rasctarg, \
                      decltarg=decltarg, \
                      
                      boolregrforc=True, \
                     )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        globals().get(sys.argv[1])(*sys.argv[2:])


