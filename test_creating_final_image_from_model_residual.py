import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib.pyplot as plt

def create_final_image(imagename,msname,final_image):
    #command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required -size 512 512 -scale 0.5arcsec -niter 10 '+\
    #                f'-name {final_image} {msname}'
    #os.system(command_str)
    
    header=fits.getheader(imagename+"-psf.fits")
    bmaj=header['BMAJ']
    bmin=header['BMIN']
    bpa=(header['BPA'])*np.pi/180
    cell=abs(header['CDELT1'])
    
    bmaj_pix=bmaj/cell
    bmin_pix=bmin/cell
    
    sigma_major_pix=bmaj_pix/(2*np.sqrt(2*np.log(2)))
    sigma_minor_pix=bmin_pix/(2*np.sqrt(2*np.log(2)))
    
    kernel=Gaussian2DKernel(sigma_major_pix,sigma_minor_pix,bpa)
    kernel=kernel.array/kernel.array.max()
    
    model_data=fits.getdata(imagename+"-model.fits")[0,0,:,:]
    smoothed=convolve(model_data,kernel,normalize_kernel=False)
    

    residual_data=fits.getdata(imagename+"-residual.fits")[0,0,:,:]
    
    final_image_data=smoothed+residual_data
    
    fig,ax=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True)
    im=ax[0].imshow(residual_data,origin='lower')
    plt.colorbar(im,ax=ax[0])
    im=ax[1].imshow(smoothed,origin='lower')
    plt.colorbar(im,ax=ax[1])
    im=ax[2].imshow(final_image_data,origin='lower')
    plt.colorbar(im,ax=ax[2])
    plt.show()
    
    #with fits.open(final_image+"-image.fits",mode='update') as hdul:
    #    hdul[0].data[0,0,:,:]=final_image_data
    #    hdul.flush()
    
    return


msname='simulated_single_point.ms'
imagename_tim='test_simulated_single_source_wsclean_backup-0000'
final_image="test_self_major_minor"
create_final_image(imagename_tim,msname,final_image)
