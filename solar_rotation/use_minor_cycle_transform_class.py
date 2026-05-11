import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time
import matplotlib.pyplot as plt
from image_plane_correction_in_minor_cycle_agnostic_shift import image_plane_correction_minor_cycle as img_corr
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

def forward_transform_image(data,imagename,ref_time):
    header=fits.getheader(imagename)
    timestamp_isot=header['DATE-OBS']
    time_obj=Time(timestamp_isot,format='isot')
    cell=abs(header['CDELT1'])*3600
    mjd=time_obj.mjd
    ref_time_mjd=ref_time.mjd
    if ref_time_mjd>mjd:
        return data
    shift=15/cell
    return np.roll(data,shift,axis=0)

def backward_transform_image(imagename,ref_time):
    raw_data=fits.getdata(imagename)[0,0,:,:]
    header=fits.getheader(imagename)
    timestamp_isot=header['DATE-OBS']
    cell=abs(header['CDELT1'])*3600
    time_obj=Time(timestamp_isot,format='isot')
    mjd=time_obj.mjd
    ref_time_mjd=ref_time.mjd
    if ref_time_mjd>mjd:
        shift=0
    else:
        shift=-15/cell ### in arcsec
    return np.expand_dims(np.roll(raw_data,shift,axis=0),axis=(0,1))

def do_masking(data):
    shape=data.shape
    half_x=shape[3]//2
    half_y=shape[2]//2
    data1=np.zeros_like(data)
    data1[...]=np.nan
    data1[0,0,half_y-10:half_y+10,half_x-10:half_x+10]=data[0,0,half_y-10:half_y+10,half_x-10:half_x+10]
    return data1
    
def deconvolve(residual, model, psf,threshold,max_iterations=100,mgain=0.2):
    nchan, npol, height, width = residual.shape
    


    # residual and model are numpy arrays with dimensions nchan x npol x height x width
    # psf is a numpy array with dimensions nchan x height x width

    # This file doesn't support multiple channels or polarizations:
    if nchan != 1 or npol != 1:
        raise NotImplementedError("nchan and npol must be one")
        

    masked_residual=do_masking(np.abs(residual))
    max_val=np.nanmax(masked_residual)
    index=np.where(np.abs(masked_residual-max_val)<1e-5)
    
    peak_value = residual[index[0][0],index[1][0],index[2][0],index[3][0]]

    mgain_threshold = abs(peak_value) * (1.0 - mgain)
    first_threshold = mgain_threshold
                    #max(meta.major_iter_threshold, meta.final_threshold, mgain_threshold)

    iteration_number=0
    while (abs(peak_value) > first_threshold and abs(peak_value)>threshold and iteration_number < max_iterations):
        print(f"peak={peak_value}, first threshold={first_threshold}")
        model[index[0][0],index[1][0],index[2][0],index[3][0]] += peak_value*mgain

        psf_shift = (index[2][0] + height // 2, index[3][0] + width // 2)
        residual = residual - peak_value*mgain * np.roll(psf, psf_shift, axis=(1, 2))
        

        masked_residual=do_masking(np.abs(residual))
        max_val=np.nanmax(masked_residual)
        index=np.where(np.abs(masked_residual-max_val)<1e-5)
        
        peak_value = residual[index[0][0],index[1][0],index[2][0],index[3][0]]
    
    
        
        

    print(f"Stopped after iteration {iteration_number}, peak={peak_value}")

    # Fill a dictionary with values that wsclean expects:
    result = dict()
    result["residual"] = residual
    result["model"] = model
    result["level"] = peak_value
    result["continue"] = False#(peak_value > meta.final_threshold and \
                            #meta.iteration_number < meta.max_iterations)

    return result
    
msname='fasr_simulated_single_source_shift_included.ms'
ref_time_isot='2020-02-01T19:04:00'

shift_cor=img_corr(forward_transform_image,\
                    backward_transform_image,\
                    deconvolve,\
                    msname,\
                    ref_time_isot)

shift_cor.imagename='test_simulated_single_source_wsclean_self'
shift_cor.final_image="test_self_major_minor_mgain_0.2"
shift_cor.max_major_cycle=20
shift_cor.threshold=0.02

shift_cor.image_with_shift_correction()
    


