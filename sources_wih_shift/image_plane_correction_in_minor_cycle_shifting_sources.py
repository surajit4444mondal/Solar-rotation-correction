import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time
import matplotlib.pyplot as plt

def transform_image(data,timestamp_isot,direction='forward'):
    time_obj=Time(timestamp_isot,format='isot')
    mjd=time_obj.mjd
    ref_time=Time('2020-02-01T19:04:00',format='isot')
    ref_time_mjd=ref_time.mjd
    if ref_time_mjd>mjd:
        return data
    if direction=='forward':
        shift=30
    elif direction=='backward':
        shift=-30
    return np.roll(data,shift,axis=0)


def get_peak_residual(imagename):
    data=fits.getdata(imagename)
    return np.nanmax(data)
    
def deconvolve(residual, model, psf,max_iterations=100,mgain=0.4):
    nchan, npol, height, width = residual.shape
    
    print (residual.shape,model.shape,psf.shape)


    # residual and model are numpy arrays with dimensions nchan x npol x height x width
    # psf is a numpy array with dimensions nchan x height x width

    # This file doesn't support multiple channels or polarizations:
    if nchan != 1 or npol != 1:
        raise NotImplementedError("nchan and npol must be one")
        

    # find the largest peak in residual
    peak_index = np.unravel_index(np.argmax(residual), residual.shape)
    peak_value = residual[peak_index]

    mgain_threshold = peak_value * (1.0 - mgain)
    first_threshold = mgain_threshold
                    #max(meta.major_iter_threshold, meta.final_threshold, mgain_threshold)

    iteration_number=0
    while (peak_value > first_threshold and iteration_number < max_iterations):
        print(f"peak={peak_value}, first threshold={first_threshold}")
        model[peak_index] += peak_value*mgain

        psf_shift = (peak_index[2] + height // 2, peak_index[3] + width // 2)
        residual = residual - peak_value*mgain * np.roll(psf, psf_shift, axis=(1, 2))
        

        peak_index = np.unravel_index(
            np.argmax(residual), residual.shape
        )
        peak_value = residual[peak_index]

        iteration_number = iteration_number + 1
        
        # find the largest peak
        peak_index = np.unravel_index(np.argmax(residual), residual.shape)
        peak_value = residual[peak_index]
        
        

    print(f"Stopped after iteration {iteration_number}, peak={peak_value}")

    # Fill a dictionary with values that wsclean expects:
    result = dict()
    result["residual"] = residual
    result["model"] = model
    result["level"] = peak_value
    result["continue"] = False#(peak_value > meta.final_threshold and \
                            #meta.iteration_number < meta.max_iterations)

    print("Finished deconvolve()")
    return result

def blank_image(imagename,data=None):
    with fits.open(imagename,mode='update') as hdul:
        hdul[0].data[...]=0.0
        hdul.flush()
    return

def copy_dirty_image_to_residual(imagename):
    dirty_data=fits.getdata(imagename+"-dirty.fits")
    imagename=imagename+"-residual.fits"
    with fits.open(imagename,mode='update') as hdul:
        hdul[0].data[...]=dirty_data
        hdul.flush()
    return

def create_final_image(imagename,msname,num_chunks,final_image):
    
    
    header=fits.getheader(final_image+"-psf.fits")
    bmaj=header['BMAJ']
    bmin=header['BMIN']
    bpa=(header['BPA']+90)*np.pi/180
    cell=abs(header['CDELT1'])
    
    bmaj_pix=bmaj/cell
    bmin_pix=bmin/cell
    
    sigma_major_pix=bmaj_pix/(2*np.sqrt(2*np.log(2)))
    sigma_minor_pix=bmin_pix/(2*np.sqrt(2*np.log(2)))
    
    kernel=Gaussian2DKernel(sigma_major_pix,sigma_minor_pix,bpa)
    kernel=kernel.array/kernel.array.max()
    
    for i in range(num_chunks):
        if i==0:
            residual_data=fits.getdata(imagename+"-"+str(i).zfill(4)+"-residual.fits")[0,0,:,:]
        else:
            residual_data+=fits.getdata(imagename+"-"+str(i).zfill(4)+"-residual.fits")[0,0,:,:]
            
    residual_data/=num_chunks
    
    model_data=fits.getdata(final_image+"-model.fits")[0,0,...]
    smoothed=convolve(model_data,kernel,normalize_kernel=False)    
    
    final_image_data=smoothed+residual_data
    
    with fits.open(final_image+"-image.fits",mode='update') as hdul:
        hdul[0].data[0,0,:,:]=final_image_data
        hdul.flush()
    
    return

def get_residual(imagename):
    raw_data=fits.getdata(imagename)[0,0,:,:]
    header=fits.getheader(imagename)
    timestamp_isot=header['DATE-OBS']
    transformed_data=transform_image(raw_data,timestamp_isot,direction='backward')  ### when going from UV plane to image plane
    
    
    return np.expand_dims(transformed_data,axis=(0,1))
    
def update_model_image(imagename,model):
    header=fits.getheader(imagename)
    timestamp_isot=header['DATE-OBS']
    transformed_data=np.expand_dims(transform_image(model[0,0,:,:],timestamp_isot,direction='forward'),axis=(0,1))  ### when going from image plane to UV plane
    
    with fits.open(imagename,mode='update') as hdul:
        hdul[0].data[...]=transformed_data
        hdul.flush()
    return
    
   
def create_dirty_image_all_times(final_image,msname):

    command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required -size 512 512 -scale 0.5arcsec -niter 10 '+\
                        f'-name {final_image} {msname}'
    os.system(command_str)    
    return
            
def do_minor_cycle(imagename,num_chunk,final_image):
    for i in range(num_chunk):
        imagename_tim=imagename+"-"+str(i).zfill(4)+"-residual.fits"
        if i==0:
            residual=get_residual(imagename_tim)
        else:
            residual+=get_residual(imagename_tim)
    residual/=num_chunk
    
    model=fits.getdata(final_image+"-model.fits")
    psf=fits.getdata(final_image+"-psf.fits")[0,...]
    
    result=deconvolve(residual, model, psf)
    
    for i in range(num_chunk):
        modelname_tim=imagename+"-"+str(i).zfill(4)+"-model.fits"
        update_model_image(modelname_tim,result['model'])
    
    update_model_image(final_image+"-model.fits",result['model'])
    
    return 
    
    
    
    
    
    
    
    
            
    
threshold=0.18
max_major_cycle=3

imagename='test_simulated_single_source_wsclean_self'
msname='fasr_simulated_single_source_shift_included.ms'
minor_cycle_function='write_minor_cycle_function.py'
final_image="test_self_major_minor"

peak_vals=[]

j=0

intervals=[[0,1],[1,2]]

create_dirty_image_all_times(final_image,msname)

command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required -size 512 512 -scale 0.5arcsec -niter 10 '+\
                    f'-name {final_image} {msname}'
os.system(command_str)

while True:
    peak_val1=[]
    for num_interval,interval in enumerate(intervals):
        imagename_tim=imagename+"-"+str(num_interval).zfill(4)
        if j==0:
            continue1=''
        else:
            continue1=' -continue'
        
        if j==0:
            ###Creating dummy image. I am using very small iter to create the basic image structures. I set the model to 0, and residual to dirty image
            ### before passing it to the minor cycle.
            command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required {continue1} -size 512 512 -scale 0.5arcsec -niter 10 -interval {interval[0]} {interval[1]} '+\
                    f'-name {imagename_tim} {msname}'
            os.system(command_str)
            blank_image(imagename_tim+"-model.fits")
            copy_dirty_image_to_residual(imagename_tim)
        else:
            ### Creating a dirty image. I only need to put the dirty image into the residual. Note that the residual already present is not corrected after the 
            ### major cycle. Hence this step is necesary.
            command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required {continue1} -size 512 512 -scale 0.5arcsec -niter 0 -interval {interval[0]} {interval[1]} '+\
                    f'-name {imagename_tim} {msname}'
            os.system(command_str)
            copy_dirty_image_to_residual(imagename_tim)
    
    
    do_minor_cycle(imagename,len(intervals),final_image)
    
    for num_interval,interval in enumerate(intervals):
        imagename_tim=imagename+"-"+str(num_interval).zfill(4)   
        command_str=f'singularity exec /data/simpl.sif wsclean --predict --no-dirty -size 512 512 -scale 0.5arcsec -interval {interval[0]} {interval[1]} '+\
                    f'-name {imagename_tim} {msname}'
        os.system(command_str)
        
        residual=imagename_tim+"-residual.fits"
        peak_val=get_peak_residual(residual)

        peak_val1.append(peak_val)
    peak_vals.append(peak_val1)
    
    if all(peak_val1)<threshold:
        break
    if j>max_major_cycle:
        break
    j+=1



create_final_image(imagename,msname,len(intervals),final_image)

print (peak_vals)
    
    
     
    
