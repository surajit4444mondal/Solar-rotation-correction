import numpy as np
import datetime as dt
import os,glob
import sys
sys.path.append('/home/surajit/envs/default_python/lib/python3.8/site-packages/')
sys.path.append('/home/surajit/envs/default_python/lib/python3.8/site-packages/suncasa-1.0.0-py3.8.egg')
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
from suncasa.utils import helioimage2fits as hf
from casatasks import split,ft,uvsub,concat,delmod
from casatools import image
from sunpy import map as smap
import astropy.units as u
from sunpy.map.maputils import coordinate_is_on_solar_disk
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from scipy import ndimage


def rotateimage(data,xc_centre,yc_centre,p_angle):
	'''
	data: fits image
	angle: NASA horizon angle
	xc,yc: in pixel centre
	Negative angle imples anti-clockwise rotation and vice versa
	'''
	padX=[data.shape[1]-xc_centre,xc_centre]
	padY=[data.shape[0]-yc_centre,yc_centre]
	imgP=np.pad(data,[padY,padX],'constant')
	imgR=ndimage.rotate(imgP,p_angle,reshape=False,order=0,prefilter=False)
	return imgR[padY[0]:-padY[1],padX[0]:-padX[1]]


def transform_image(imagename,timerange,msname,tdt,reftime):
	
	#imagename='eovsa_2336_image.model'
	helio_image=imagename[:-6]+"_helio_model.fits"
	
	hf.imreg(vis=msname,imagefile=imagename,timerange=timerange,fitsfile=helio_image,\
			usephacenter=False,verbose=True)
	
	
	helio=hf.ephem_to_helio(vis='../UDB20210630_noselfcal.ms',reftime=reftime.strftime("%Y/%m/%d/%H:%M:%S"))
	p0=helio[0]['p0']
	
	eomap=smap.Map(helio_image)
	eotime=eomap.date
	img_data=eomap.data
	del img_data
	
	meta=eomap.meta

	naxis=meta['naxis1']
	
	ia=image()
	ia.open(imagename)
	data=ia.getchunk()[:,:,0,0]
	ia.close()
	img_data=rotateimage(data.T,naxis//2,naxis//2,-p0)

	x=np.arange(0,naxis,1)


	xrot=np.zeros(int(naxis*naxis))
	yrot=np.zeros_like(xrot)
	#values=np.zeros_like(xrot)
	image_rot=np.zeros_like(img_data)
	i=0

	pos=np.where(abs(img_data)>1e-3)
	print (eomap.meta['date-obs'],tdt)
	for y1,x1 in zip(pos[0],pos[1]):
		coord=eomap.pixel_to_world(x1*u.pix,y1*u.pix)
		if coordinate_is_on_solar_disk(coord)==True:
			rotated_coord = solar_rotate_coordinate(coord, time=tdt)
			coord_new=eomap.world_to_pixel(rotated_coord)
			xrot[i]=coord_new.x.value
			yrot[i]=coord_new.y.value
			#print (xrot[i],x1)
			
			
		else:
			xrot[i]=x1
			yrot[i]=y1
		
		image_rot[int(yrot[i]),int(xrot[i])]=img_data[y1,x1]
		i+=1

	
	
	pos=np.where(np.isnan(image_rot)==True)
	image_rot[pos]=0

	new_img_data=rotateimage(image_rot,naxis//2,naxis//2,p0)   ### rotating it back to original ra-dec frame
	
	#X,Y=np.meshgrid(x,x)
	#plt.imshow(data.T,origin='lower')
	#plt.contour(X,Y,new_img_data,levels=np.array([0.2,0.4,0.6,0.8])*329,colors='r')
	#plt.show()

	#os.system("cp -r "+imagename+" temp1.model")
	#imagename='temp1.model'
	ia=image()
	ia.open(imagename)
	data=ia.getchunk()
	data[:,:,0,0]=new_img_data.T
	ia.putchunk(data)
	ia.close()
	#raise RuntimeError
	return

def correct_shift(starttime,endtime,tdt,phrase=''):
	j=0
	st=starttime
	standard_time=starttime#dt.datetime(2021,6,30,15,36,0)
	while st<endtime:
		if j!=0:
			
			end=st+tdt
			timerange=st.strftime("%Y/%m/%d/%H:%M:%S")+\
			  	 "~"+end.strftime("%Y/%m/%d/%H:%M:%S")
			msname='eovsa_'+st.strftime("%H%M")+".ms"
			if os.path.isdir(msname)==False:
				st+=tdt
				j+=1
				continue
			imagename='eovsa_'+st.strftime("%H%M")+"_image"+phrase
			if os.path.isdir(imagename+".image")==False:
				tclean(vis=msname,imagename=imagename,imsize=[512],cell='2arcsec',niter=800,\
						interactive=True,uvrange='>1klambda',stokes='XX')
			
			imagename=imagename+".model"
			os.system("rm -rf temp.*")
			os.system("cp -r "+msname+" temp.ms")
			

			ft(vis='temp.ms',model=imagename,usescratch=True)
			uvsub(vis='temp.ms')  #### now corrected data is the purely the residual
			delmod(vis='temp.ms',scr=True)
			
			os.system("cp -r "+imagename+" temp.model")
			
			if st>standard_time:
				dt1=TimeDelta((st-standard_time).seconds,format='sec')
			else:
				dt1=TimeDelta((standard_time-st).seconds,format='sec')
				dt1=-dt1
			transform_image("temp.model",timerange=timerange,msname=msname,tdt=-dt1,reftime=st)
			
			ft(vis='temp.ms',model="temp.model",usescratch=True)
			uvsub(vis='temp.ms',reverse=True)   #### Now the corrected data column will add the residuals +
					     #### visibilities correspodning to the shifted image
			outms=msname[:-3]+"_shift_corrected.ms"
			if os.path.isdir(outms)==True:
				os.system("rm -rf "+outms)
			
			split(vis='temp.ms',outputvis=outms,datacolumn='corrected')
			
			
		else:
			msname='eovsa_'+st.strftime("%H%M")+".ms"
			outms=msname[:-3]+"_shift_corrected.ms"
			if os.path.isdir(outms)==True:
				os.system("rm -rf "+outms)			
			os.system("cp -r "+msname+" "+outms)	
			
		st+=tdt
		j+=1
	return
	

def shift(starttime,endtime,tdt,model,phrase=''):
	j=0
	st=starttime
	standard_time=starttime#dt.datetime(2021,6,30,19,0,0)
	while st<endtime:
		if j!=0:
			end=st+tdt
			timerange=starttime.strftime("%Y/%m/%d/%H:%M:%S")+\
			  	 "~"+endtime.strftime("%Y/%m/%d/%H:%M:%S")
			msname='eovsa_'+st.strftime("%H%M")+".ms"
			imagename='eovsa_'+st.strftime("%H%M")+"_image"+phrase
			
			startmodel="temp_"+st.strftime("%H%M")+".model"
			
			os.system("rm -rf "+startmodel)
			os.system("cp -r "+model+" "+startmodel)
			
			
			if st>standard_time:
				dt1=TimeDelta((st-standard_time).seconds,format='sec')
			else:
				dt1=TimeDelta((standard_time-st).seconds,format='sec')
				dt1=-dt1
			transform_image(startmodel,timerange=timerange,msname='../UDB20210630_noselfcal.ms',tdt=dt1,reftime=st)
			
			if os.path.isdir(msname)==True:
				tclean(vis=msname,imagename=imagename,imsize=[512],cell='2arcsec',niter=800,\
					interactive=True,startmodel=startmodel,uvrange='>1klambda',stokes='XX')
				
		st+=tdt
		j+=1
	return
	
def break_ms_with_time(msname,spw,starttime,endtime,tdt):
	st=starttime
	while st<endtime:
		end=st+tdt
		timerange=st.strftime("%Y/%m/%d/%H:%M:%S")+\
			   "~"+end.strftime("%Y/%m/%d/%H:%M:%S")
		outputvis='eovsa_'+st.strftime("%H%M")+".ms"
		try:
			split(vis=msname,timerange=timerange,spw=spw,\
				datacolumn='data',correlation='XX',\
				outputvis=outputvis)
		except (RuntimeError,OSError):
			pass
		st=end
		

msname='../UDB20210630_noselfcal.ms'
spw='10'
starttime=dt.datetime(2021,6,30,15,36,21)
endtime=dt.datetime(2021,6,30,20,30,22)
tdt=dt.timedelta(seconds=1200)
duration=1200

break_ms_with_time(msname,spw,starttime,endtime,tdt)


combined_vis='eovsa_shift_corrected_combined.ms'

correct_shift(starttime,endtime,tdt)

files=glob.glob("*_shift_corrected.ms")

if os.path.isdir(combined_vis)==True:
	os.system("rm -rf "+combined_vis)

concat(vis=files,concatvis=combined_vis)

tclean(vis=combined_vis,imagename='combined_image_temp',imsize=[512],cell='2arcsec',\
	niter=800,interactive=True,uvrange='>1klambda',stokes='XX')

shift(starttime,endtime,tdt,"combined_image_temp.model",phrase='1')

correct_shift(starttime,endtime,tdt,phrase='1')

files=glob.glob("*_shift_corrected.ms")

if os.path.isdir(combined_vis)==True:
	os.system("rm -rf "+combined_vis)
	
concat(vis=files,concatvis=combined_vis)

tclean(vis=combined_vis,imagename='combined_image_temp_1',imsize=[512],cell='2arcsec',\
	niter=800,interactive=True,uvrange='>1klambda',stokes='XX')

