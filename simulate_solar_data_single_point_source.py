import h5py
import numpy as np
from astropy.io import fits
from casatools import image,simulator,measures,quanta
import os
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def makeimage(ra,dec,freq,imagename='sim_onepoint_true.im'):  
	'''
	ra dec is center of image. Give in radians
	freq in GHz
	cell : e.g. '2arcsec'
	'''
	shape=512
	x=np.arange(0,shape,1)
	X,Y=np.meshgrid(x,x)
	sigmax=4
	sigmay=4
	cell=0.5
	flux=1./(2*np.pi*sigmax*sigmay)*np.exp(-0.5*((X-shape//2-15)**2/sigmax**2+(Y-shape//2)**2/sigmay**2))
	flux[flux<0.001]=0.0
	qa=quanta()
	ia=image()
	## Make the image from a shape
	ia.close()
	ia.fromshape(imagename,[shape,shape,1,1],overwrite=True)
	## Make a coordinate system
	cs=ia.coordsys()
	cs.setunits(['rad','rad','','Hz'])
	cell_rad=qa.convert(qa.quantity(str(cell)+"arcsec"),"rad")['value']
	cs.setincrement([-cell_rad,cell_rad],'direction')
	cs.setreferencevalue([ra,dec],type="direction")
	cs.setreferencevalue(str(freq)+'GHz','spectral')
	cs.setreferencepixel([0],'spectral')
	cs.setincrement('0.3GHz','spectral')
	## Set the coordinate system in the image
	ia.setcoordsys(cs.torecord())
	ia.setbrightnessunit("Jy/pixel")
	ia.set(0.0)
	ia.close()
	
		
	ia.open(imagename)
	data=ia.getchunk()
	data[:,:,0,0]=flux.T
	#plt.imshow(flux1);plt.colorbar();plt.show()
	ia.putchunk(data)
	ia.close()
	


def generate_ms(config_file,source_ra,source_dec,reftime,integration_time=60,msname='fasr.ms',duration=None):
	'''
	config_file: Antenna configuration file in standard format. 
		     First column: x
		     Second COlumn:y
		     Third Column: z
		     Fourth column: dish diameter
		     Fifth column: Antenna name
	spws:  Frequencies of the spws
	source_ra, source_dec: ra, dec of phasecenter in radians
	reftime: Reference time of observation in CASA format. in UTC 
	integration_time: in seconds
	duration: in seconds
	'''
	
	antenna_params=np.genfromtxt(config_file,usecols=(0,1,2,3))
	ant_names=np.genfromtxt(config_file,usecols=(4))
	x=antenna_params[:,0]
	y=antenna_params[:,1]
	z=antenna_params[:,2]
	dish_dia=antenna_params[:,3]
	
	
	
	sm=simulator()
	me=measures()
	sm.open(msname)
	
	sm.setconfig(telescopename="EVLA",x=x,y=y,z=z,dishdiameter=dish_dia,\
		mount='alt-az',antname=ant_names,coordsystem='global')
	
	
	sm.setspwindow(spwname='Band'+str(0),freq=str(12)+"GHz",deltafreq='100MHz',\
			freqresolution='100MHz',nchannels=1,stokes='RR LL')
		
	sm.setfeed('perfect R L')
	sm.setfield(sourcename='Sun',sourcedirection=['J2000',str(source_ra)+"rad",str(source_dec)+"rad"])
	sm.setauto(autocorrwt=0.0)
	sm.settimes(integrationtime=str(integration_time)+"s",referencetime=me.epoch('UTC',reftime),usehourangle=False)
	
	if duration==None:
		duration=integration_time
	
	starttime=str(-duration/2)+"s"
	endtime=str(duration/2)+"s"
	
	sm.observe("Sun","Band0",starttime=starttime,stoptime=endtime)
	
	
	sm.setdata(spwid=0)
	makeimage(source_ra,source_dec,12,imagename='solar_image.im')
	sm.predict(imagename='solar_image.im')
	#sm.setnoise(mode='simplenoise', simplenoise='0.16Jy')
	#sm.corrupt()
	#break
	#os.system("rm -rf solar_image.im")
	sm.close()
	
### taking solar coords from VLA data of same day. 
solar_ra=5.489929 
solar_dec=-0.30076

config_file='/home/surajit/casa-pipeline-release-5.6.3-19.el7/data/alma/simmos/vla.c.cfg'




reftime='2020/02/01/19:03:00'
duration=60

generate_ms(config_file,solar_ra,solar_dec,reftime,msname='vla_c_simulated_point_shifted.ms')



