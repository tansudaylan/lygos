from util import *
import h5py

pathdata = os.environ['TCAT_DATA_PATH'] + '/tesspsfn/'
pathtemp = pathdata + 'temp.txt'

indxcams = range(1, 5)
indxccds = range(1, 5)
indxrows = [1, 513, 1025, 1536, 2048]
indxcols = [45, 557, 1069, 1580, 2092]

#boolplotflgt = False
boolplotflgt = True

pathsave = pathdata + 'listpsfn.h5'

if os.path.exists(pathsave):
    print 'Reading from %s...' % pathsave
    objth5py = h5py.File(pathsave, 'r')
    listpsfn = objth5py.get('listpsfn')
    listpsfn = np.array(listpsfn)
    objth5py.close()
else:
    listpsfn = np.empty((4, 4, 5, 5, 117, 117))

for a in indxcams:
    for b in indxccds:
        pathsubs = pathdata + 'tess_prf-master/cam%d_ccd%d/' % (a, b)
        for k in range(len(indxrows)):
            for l in range(len(indxcols)):
                if not os.path.exists(pathsave):
                    listpsfn[a-1, b-1, k, l, :, :] = read_psfntess(a, b, indxrows[k], indxcols[l])
                
                if boolplotflgt:
                    figr, axis = plt.subplots(figsize=(12, 12))
                    axis.set_ylabel('$y$')
                    axis.set_xlabel('$x$')
                    axis.set_title('Camera %d, CCD %d, Row %d, Column %d' % (a, b, indxrows[k], indxcols[l]))
                    plt.imshow(listpsfn[a-1, b-1, k, l, :, :], cmap='Greys_r', interpolation='none')
                    plt.tight_layout()
                    pathimag = pathdata + 'psfn_%d%d%d%d.pdf' % (a, b, k, l)
                    print 'Writing to %s...' % pathimag
                    print 
                    plt.savefig(pathimag)
                    plt.close()

if not os.path.exists(pathsave):
    print 'Writing to %s...' % pathsave
    objth5py = h5py.File(pathsave, 'w')
    objth5py.create_dataset('listpsfn', data=listpsfn)
    objth5py.close()


#listline = open(pathdata + 'tesspsfn.txt', 'r')
##id|resolution|camera_id|ccd_id|stellar_type|stellar_temp|position|angle
#listpsid = []
#for k, line in enumerate(listline):
#    if k < 30:
#        print 'k'
#        print k
#        print 'line'
#        print line
#        print
#    if k < 2:
#        continue
#        
#    listcols = line.split('|')
#    listpsid.append(listcols[0])
#
#listpsid = np.array(listpsid).astype(int)
#listline.close() 
#
#summgene(listpsid)
#
#psfn = [[] for k in range(1)]
#psfn[0] = np.empty((187, 187, 9126))
##psfn[1] = np.empty((17, 17, 9126))
#
#os.system('mkdir -p %s' % pathdata) 
#numbpsfn = 9126
#numbpsfn = 9126
#indxpsfn = np.arange(numbpsfn)
#psid = indxpsfn + 1
#temp = np.empty_like(psid)
#posi = np.empty((2, numbpsfn))
#angl = np.empty((2, numbpsfn))
#
#print 'numbpsfn'
#print numbpsfn
#indxpsfngood = indxpsfn
##indxpsfngood = np.random.choice(indxpsfn, size=30)
#
#for k in indxpsfngood:
#    
#    cmnd = 'tsig-psf --id %d --show-contents > %stemp.txt' % (psid[k], pathdata)
#    os.system(cmnd)   
#    print 'k'
#    print k
#    
#    datatemp = np.loadtxt(pathtemp, skiprows=8, delimiter=',')
#    if datatemp.shape[0] == 187:
#        indxreso = 0
#    else:
#        continue
#
#    with open(pathtemp, 'r') as listline:
#        for t, line in enumerate(listline):
#            print line
#            if t == 1:
#               psidtemp = int(line.split('id=')[1]) 
#            if t == 2:
#               reso = int(line.split('resolution=')[1]) 
#            if t == 4:
#               temp[k] = int(line.split('stellar_temp=')[1]) 
#            if t == 5:
#               posi[:, k] = line.split('field_position=')[1].split('(')[1].split(')')[0].split(',')
#               posi[:, k] = [float(posi[0, k]), float(posi[1, k])]
#            if t == 6:
#               angl[:, k] = line.split('field_angle=')[1].split('(')[1].split(')')[0].split(',')
#               angl[:, k] = [float(angl[0, k]), float(angl[1, k])]
#            if t == 8:
#                break
#    
#    
#    if temp[k] == 6030 and reso == 11:
#        
#        print 'Plotting...'
#        print 'psfn[indxreso]'
#        summgene(psfn[indxreso])
#        print 'psidtemp'
#        print psidtemp
#        print 'temp[k]'
#        print temp[k]
#        print 'posi[:, k]'
#        print posi[:, k]
#        print 'angl[:, k]'
#        print angl[:, k]
#        
#        psfn[0][:, :, k] = datatemp 
#        
#        figr, axis = plt.subplots()
#        
#        axis.set_ylabel('$y$')
#        axis.set_xlabel('$x$')
#        axis.set_title('$T=%d, x=%.3g, y=%.3g$' % (temp[k], angl[0, k], angl[1, k]))
#        
#        plt.imshow(psfn[indxreso][:, :, k], cmap='Greys_r', interpolation='none')
#        plt.tight_layout()
#        pathimag = pathdata + 'psfn_fram%04d.png' % (k)
#        print 'Writing to %s...' % pathimag
#        print 
#        plt.savefig(pathimag)
#        plt.close()
#
#cmnd = 'convert -delay 20 -density 200x200 %spsfn_fram*.png %spsfn.gif' % (pathdata, pathdata)
#print cmnd
#os.system(cmnd)
#

            #import scipy
            #from scipy.signal import lombscargle
            ## PSF difference
            #figr, axis = plt.subplots(figsize=(20, 6))
            #axis.set_ylabel('LS')
            #axis.set_xlabel('Frequency [1/days]')
            #for a in range(2):
            #    if a == 0:
            #        ydat = (gdat.lcuraperdiff[:, 0, 2, 1] - gdat.lcuraperdiff[:, 0, 2, 2])
            #        labl = 'x'
            #    else:
            #        ydat = (gdat.lcuraperdiff[:, 0, 2, 3] - gdat.lcuraperdiff[:, 0, 2, 4])
            #        labl = 'y'
            #    ydat -= np.mean(ydat)
            #    ydat /= gdat.lcuraperdiff[:, 0, 2, 0]
            #    ydat *= 100.
            #    ydat = scipy.signal.lombscargle(gdat.timedata, ydat, np.linspace(0.01, 0.5, 1000))
            #    axis.plot(np.linspace(0.01, 0.5, 1000), ydat, label=labl, ls='', marker='o', markersize=5, alpha=0.3)
            #axis.legend()
            #plt.tight_layout()
            #path = gdat.pathdata + 'ffftpsfn_%s.png' % (gdat.strgsaveextn)
            #print 'Writing to %s...' % path
            #plt.savefig(path)
            #plt.close()
        
            # PSF difference
            #figr, axis = plt.subplots(figsize=(20, 6))
            #axis.set_ylabel('Diff [%]')
            #axis.set_xlabel('Time since %s [days]' % objttimeinit.iso)
            #for a in range(2):
            #    if a == 0:
            #        ydat = (gdat.lcuraperdiff[:, 0, 2, 1] - gdat.lcuraperdiff[:, 0, 2, 2])
            #        labl = 'x'
            #    else:
            #        ydat = (gdat.lcuraperdiff[:, 0, 2, 3] - gdat.lcuraperdiff[:, 0, 2, 4])
            #        labl = 'y'
            #    ydat -= np.mean(ydat)
            #    ydat /= gdat.lcuraperdiff[:, 0, 2, 0]
            #    ydat *= 100.
            #    axis.plot(gdat.timedata, ydat, label=labl, ls='', marker='o', markersize=5, alpha=0.3)
            #axis.legend()
            #axis.set_ylim([-100, 100])
            #plt.tight_layout()
            #path = gdat.pathdata + 'lcurpsfn_%s.png' % (gdat.strgsaveextn)
            #print 'Writing to %s...' % path
            #plt.savefig(path)
            #plt.close()
  
        

