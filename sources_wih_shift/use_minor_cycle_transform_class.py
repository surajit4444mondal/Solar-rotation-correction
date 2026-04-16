import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time
import matplotlib.pyplot as plt
from image_plane_correction_in_minor_cycle_agnostic_shift import image_plane_correction_minor_cycle as img_corr

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
    
def deconvolve(residual, model, psf,threshold,max_iterations=50,mgain=0.1):
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
shift_cor.max_major_cycle=30
shift_cor.threshold=0.005
shift_cor.do_continue=True

shift_cor.image_with_shift_correction()
    


