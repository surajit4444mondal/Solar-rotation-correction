import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time
import matplotlib.pyplot as plt

    
class image_plane_correction_minor_cycle():
    def __init__(self,forward_transform,backward_transform,deconvolution_function,msname,ref_time_isot):
        self.forward_transform=forward_transform
        self.backward_transform=backward_transform
        self.msname=msname
        self.ref_time=Time(ref_time_isot,format='isot')    
        self.threshold=0.18
        self.max_major_cycle=3
        self.intervals=[[0,1],[1,2]]
        self.imagename='test_simulated_single_source_wsclean_self'
        self.final_image="test_self_major_minor"
        self.deconvolution_function=deconvolution_function
        self.imsize=512
        self.cell=0.5  ###arcsec
        self.do_continue=False

        
    def get_residual(self,imagename):
        transformed_data=self.backward_transform(imagename,self.ref_time)  ### when going from UV plane to image plane
        return transformed_data
    
    def update_model_image(self,imagename,model):
        
        transformed_data=np.expand_dims(self.forward_transform(model[0,0,:,:],imagename,self.ref_time),axis=(0,1))  ### when going from image plane to UV plane
        
        with fits.open(imagename,mode='update') as hdul:
            hdul[0].data[...]=transformed_data
            hdul.flush()
        return
    
    def create_dirty_image_all_times(self):

        command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required -size {self.imsize} {self.imsize} -scale {self.cell}arcsec -niter 10 '+\
                            f'-name {self.final_image} {self.msname}'
        os.system(command_str)    
        return
    
    
    def image_with_shift_correction(self):
        
        peak_vals=[]

        j=0

        if not self.do_continue:
            self.create_dirty_image_all_times()

            command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required -size {self.imsize} {self.imsize} -scale {self.cell}arcsec -niter 10 '+\
                                f'-name {self.final_image} {self.msname}'
            os.system(command_str)
            self.blank_image(self.final_image+"-model.fits")

        while True:
            peak_val1=[]
            for num_interval,interval in enumerate(self.intervals):
                imagename_tim=self.imagename+"-"+str(num_interval).zfill(4)
                if j==0 and not self.do_continue:
                    continue1=''
                else:
                    continue1=' -continue'
                
                if j==0 and not self.do_continue:
                    ###Creating dummy image. I am using very small iter to create the basic image structures. I set the model to 0, and residual to dirty image
                    ### before passing it to the minor cycle.
                    command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required {continue1} -size {self.imsize} {self.imsize} -scale {self.cell}arcsec '+\
                                f' -niter 10 -interval {interval[0]} {interval[1]} -name {imagename_tim} {self.msname}'

                    
                    os.system(command_str)
                    self.blank_image(imagename_tim+"-model.fits")
                    self.copy_dirty_image_to_residual(imagename_tim)
                else:
                    ### Creating a dirty image. I only need to put the dirty image into the residual. Note that the residual already present is not corrected after the 
                    ### major cycle. Hence this step is necesary.
                    command_str=f'singularity exec /data/simpl.sif wsclean -no-update-model-required {continue1} -size {self.imsize} {self.imsize} -scale {self.cell}arcsec '+\
                                    f'-niter 0 -interval {interval[0]} {interval[1]} -name {imagename_tim} {self.msname}'
                    
                    os.system(command_str)
                    self.copy_dirty_image_to_residual(imagename_tim)
            
            #fig,ax=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True)
            #for num_interval,interval in enumerate(self.intervals):
            #    data=fits.getdata(self.imagename+"-"+str(num_interval).zfill(4)+"-residual.fits")
            #    im=ax[num_interval].imshow(data[0,0,:,:],origin='lower')
            #    plt.colorbar(im,ax=ax[num_interval])
            
            
            
            max_residual_value,residual=self.do_minor_cycle()
            
            #im=ax[2].imshow(residual[0,0,:,:],origin='lower')
            #plt.colorbar(im,ax=ax[2])
            #plt.show()
            
            for num_interval,interval in enumerate(self.intervals):
                imagename_tim=self.imagename+"-"+str(num_interval).zfill(4)   
                command_str=f'singularity exec /data/simpl.sif wsclean --predict --no-dirty -size {self.imsize} {self.imsize} -scale {self.cell}arcsec -interval {interval[0]} {interval[1]} '+\
                            f'-name {imagename_tim} {self.msname}'


                os.system(command_str)
                
                residual=imagename_tim+"-residual.fits"
                peak_val=self.get_peak_residual(residual)


                peak_val1.append(peak_val)
            
            peak_vals+=peak_val1
            
            print ("Peak value of residuals:",peak_val1)
            if max(peak_val1)<self.threshold or max_residual_value<self.threshold:
                break
            if j>self.max_major_cycle:
                break
            j+=1
        self.create_final_image()
        
            
    def do_minor_cycle(self):
        num_chunk=len(self.intervals)
        for i in range(num_chunk):
            imagename_tim=self.imagename+"-"+str(i).zfill(4)+"-residual.fits"
            if i==0:
                residual=self.get_residual(imagename_tim)
            else:
                residual+=self.get_residual(imagename_tim)
        residual/=num_chunk
        
        model=fits.getdata(self.final_image+"-model.fits")
        psf=fits.getdata(self.final_image+"-psf.fits")[0,...]
        
        result=self.deconvolution_function(residual, model, psf,self.threshold)
        
        for i in range(num_chunk):
            modelname_tim=self.imagename+"-"+str(i).zfill(4)+"-model.fits"
            self.update_model_image(modelname_tim,result['model'])
        

        self.update_model_image(self.final_image+"-model.fits",result['model'])
        
        return np.nanmax(np.abs(result['residual'])),result['residual']
    
    def create_final_image(self):
        num_chunks=len(self.intervals)
        header=fits.getheader(self.final_image+"-psf.fits")
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
                residual_data=fits.getdata(self.imagename+"-"+str(i).zfill(4)+"-residual.fits")[0,0,:,:]
            else:
                residual_data+=fits.getdata(self.imagename+"-"+str(i).zfill(4)+"-residual.fits")[0,0,:,:]
                
        residual_data/=num_chunks
        
        model_data=fits.getdata(self.final_image+"-model.fits")[0,0,...]
        smoothed=convolve(model_data,kernel,normalize_kernel=False)    
        
        final_image_data=smoothed+residual_data
        
        with fits.open(self.final_image+"-image.fits",mode='update') as hdul:
            hdul[0].data[0,0,:,:]=final_image_data
            hdul.flush()
        
        return

    @staticmethod    
    def blank_image(imagename,data=None):
        with fits.open(imagename,mode='update') as hdul:
            hdul[0].data[...]=0.0
            hdul.flush()
        return
    
    @staticmethod
    def copy_dirty_image_to_residual(imagename):
        dirty_data=fits.getdata(imagename+"-dirty.fits")
        imagename=imagename+"-residual.fits"
        with fits.open(imagename,mode='update') as hdul:
            hdul[0].data[...]=dirty_data
            hdul.flush()
        
        
        return
    
    @staticmethod    
    def get_peak_residual(imagename):
        data=fits.getdata(imagename)
        return np.nanmax(data)
        

    
    
    
    
    
    
            
    









    
    
     
    
