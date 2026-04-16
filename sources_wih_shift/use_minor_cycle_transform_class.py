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

    
msname='fasr_simulated_single_source_shift_included.ms'
ref_time_isot='2020-02-01T19:04:00'
maskfile='shift_corrected_mask.npy'

shift_cor=img_corr(forward_transform_image,\
                    backward_transform_image,\
                    msname,\
                    ref_time_isot,\
                    maskfile=maskfile)

shift_cor.imagename='test_simulated_single_source_wsclean_self'
shift_cor.final_image="test_masking"
shift_cor.max_major_cycle=10
shift_cor.threshold=0.05
shift_cor.do_continue=True

shift_cor.interactive=False

shift_cor.image_with_shift_correction()
    


