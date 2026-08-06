import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time
import matplotlib.pyplot as plt
from casatools import table
from .masking_utils import MaskingSelector

def remove_column(msname):
    tb=table()
    tb.open(msname,nomodify=False)
    try:
        colnames=tb.colnames()
        if 'IMAGING_WEIGHT_SPECTRUM' in colnames:
            tb.removecols('IMAGING_WEIGHT_SPECTRUM')
            tb.flush()
    finally:
        tb.close()
    return
 
def update_weight_column(msname,initialise=False):
    tb=table()
    tb.open(msname,nomodify=False)
    try:
        
        weights=tb.getcol('WEIGHT')
        if not initialise:
            imaging_weights=np.mean(tb.getcol('IMAGING_WEIGHT_SPECTRUM'),axis=1)
        else:
            imaging_weights=np.ones_like(weights)
        
        tb.putcol('WEIGHT',imaging_weights)
        tb.flush()
    finally:
        tb.close()
    return
    
def run_wsclean(container_path, msname, options, predict=False):
    """Generic wrapper for shell-based WSClean calls."""
    base_cmd = f"singularity exec {container_path} wsclean"
    args = " ".join([f"-{k} {v}" if v != "" else f"-{k}" for k, v in options.items()])
    #args+=' -minuv-l 100'
    if predict:
        args=args+' --predict'
    command = f"{base_cmd} {args} {msname} > /dev/null"  ### redirecting prints to null
    os.system(command)

    
class image_plane_correction_minor_cycle():
    def __init__(self,forward_transform,backward_transform,msname,ref_time_isot,intervals,maskfile=None):
        self.forward_transform=forward_transform
        self.backward_transform=backward_transform
        self.msname=msname
        self.ref_time=Time(ref_time_isot,format='isot')    
        
        # Configuration parameters
        self.settings = {
            'imsize': 512,
            'cell': '0.5arcsec',
            'threshold': 0.18,
            'mgain': 0.1,
            'max_major_cycle': 3,
            'container': '/data/simpl.sif',
            'pol':'I',
            'continue': False,
            'max_iterations':50,
            'weight':'uniform'     
        }
        self.interactive=True
        self.intervals=intervals
        self.imagename='test_simulated_single_source_wsclean_self'
        self.final_image="test_self_major_minor"
        self.image_normalisers=[None]*len(self.intervals)
        if maskfile:
            self.mask=np.load(maskfile)
        self.mask_class=MaskingSelector
        
        
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
        opts = {
            'size': f"{self.settings['imsize']} {self.settings['imsize']}",
            'scale': self.settings['cell'],
            'weight': f"{self.settings['weight']}",
            'niter': 10,
            'name': self.final_image,
            'store-imaging-weights': '',
            'no-update-model-required': '',
            'pol':f"{self.settings['pol']}"
        }
                            
        run_wsclean(self.settings['container'], self.msname, opts)
        return
        
    def run_initial_synthesis(self):
        self.create_dirty_image_all_times()
        self.blank_image(self.final_image+"-model.fits")
        self.update_image_time()
        update_weight_column(self.msname)
    
    def update_image_time(self):
        for img in ['image','model']:
            with fits.open(self.final_image+f"-{img}.fits",mode='update') as hdul:
                hdul[0].header['DATE-OBS']=self.ref_time.isot
                hdul.flush()
    
    
    def image_time_chunks(self, iteration, do_continue):
        for num_interval,interval in enumerate(self.intervals):
            imagename_tim=self.imagename+"-"+str(num_interval).zfill(4)
            
            opts={
                    'no-update-model-required':'',
                    'size': f"{self.settings['imsize']} {self.settings['imsize']}",
                    'scale': self.settings['cell'],
                    'weight': 'natural',
                    'niter': 0,
                    'name': imagename_tim,
                    'pol':f"{self.settings['pol']}",
                    '-use-weights-as-taper':'',
                    'interval': f"{interval[0]} {interval[1]}"
                }
            
            if iteration==0:
                opts['save-weights']=''
                opts['niter']=10
                
            if iteration!=0 or do_continue:
                opts['continue']=''
                opts['niter']=0
            
            run_wsclean(self.settings['container'], self.msname, opts)
            
            self.copy_dirty_image_to_residual(imagename_tim)
            
            if iteration==0:
                self.blank_image(imagename_tim+"-model.fits")
                uv_weight_data=fits.getdata(imagename_tim+"-weights.fits")
                self.image_normalisers[num_interval]=np.sum(uv_weight_data)
        
    
    def perform_imaging(self):
        if not self.settings['continue']:
            remove_column(self.msname)  ### I just remove the imaging_weight column
            update_weight_column(self.msname,initialise=True)
            self.run_initial_synthesis()
        

        for j in range(self.settings['max_major_cycle']):
            print(f"--- Starting Major Cycle {j} ---")
            
            # 1. Update chunks
            self.image_time_chunks(iteration=j, do_continue=self.settings['continue'])
            
            # 2. Minor Cycle (Deconvolution)
            max_residual_value=self.do_minor_cycle()
            
            # 3. Predict/Update model back to UV plane
            self.predict_model_to_ms()

            if max_residual_value < self.settings['threshold']:
                print("Convergence reached.")
                break
        self.image_time_chunks(iteration=j+1, do_continue=self.settings['continue'])
        self.create_final_image()
        
        
    
    def predict_model_to_ms(self):
        for num_interval,interval in enumerate(self.intervals):
            imagename_tim=self.imagename+"-"+str(num_interval).zfill(4)
            opts={
                    'size': f"{self.settings['imsize']} {self.settings['imsize']}",
                    'scale': self.settings['cell'],
                    'weight': 'natural',
                    'niter': 0,
                    'name': imagename_tim,
                    'pol':f"{self.settings['pol']}",
                    '-use-weights-as-taper':'',
                    'interval': f"{interval[0]} {interval[1]}",
                }
            
            run_wsclean(self.settings['container'], self.msname, opts,predict=True)
            
            
    def do_minor_cycle(self):
        num_chunk=len(self.intervals)
        for i in range(num_chunk):
            imagename_tim=self.imagename+"-"+str(i).zfill(4)+"-residual.fits"
            if i==0:
                residual=self.get_residual(imagename_tim)*self.image_normalisers[i]
                
            else:
                residual+=self.get_residual(imagename_tim)*self.image_normalisers[i]
        residual/=np.sum(self.image_normalisers)
        
        residual_data=residual.squeeze()

        
       
        model=fits.getdata(self.final_image+"-model.fits")
        psf=fits.getdata(self.final_image+"-psf.fits")[0,...]
        
        result=self.deconvolve(residual, model, psf,self.settings['threshold'])
        
        for i in range(num_chunk):
            modelname_tim=self.imagename+"-"+str(i).zfill(4)+"-model.fits"
            self.update_model_image(modelname_tim,result['model'])
        
        self.update_model_image(self.final_image+"-model.fits",result['model'])
        
        return np.nanmax(np.abs(result['residual']))
    
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
                residual_data=fits.getdata(self.imagename+"-"+str(i).zfill(4)+"-residual.fits")[0,0,:,:]*self.image_normalisers[i]
            else:
                residual_data+=fits.getdata(self.imagename+"-"+str(i).zfill(4)+"-residual.fits")[0,0,:,:]*self.image_normalisers[i]
                
        residual_data/=np.sum(self.image_normalisers)
        
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
    
    def deconvolve(self,residual, model, psf,threshold):
        nchan, npol, height, width = residual.shape
        


        # residual and model are numpy arrays with dimensions nchan x npol x height x width
        # psf is a numpy array with dimensions nchan x height x width

        # This file doesn't support multiple channels or polarizations:
        if nchan != 1 or npol != 1:
            raise NotImplementedError("nchan and npol must be one")
        
        
        if self.interactive:
            create_mask=self.mask_class(residual[0,0,:,:])
            plt.show()
            
            self.mask=np.expand_dims(create_mask.full_mask,axis=(0,1))
            self.interactive=create_mask.interactive
        elif not hasattr(self,'mask'):
            self.mask=np.ones(residual.shape,dtype=bool)
        elif self.mask.shape!=residual.shape:
            self.mask=np.expand_dims(self.mask,axis=(0,1))
            if self.mask.shape!=residual.shape:
                raise RuntimeError("Shape of provided mask does not match")
            
            
        

        masked_residual=self.do_masking(residual,self.mask)
        max_val=np.nanmax(masked_residual)
        index=np.where(np.abs(masked_residual-max_val)<1e-5)
        
        peak_value = residual[index[0][0],index[1][0],index[2][0],index[3][0]]

        mgain_threshold = abs(peak_value) * (1.0 - self.settings['mgain'])
        first_threshold = mgain_threshold
        
        

        iteration_number=0
        while (abs(peak_value) > first_threshold and abs(peak_value)>threshold and iteration_number < self.settings['max_iterations']):
            print(f"peak={peak_value}, first threshold={first_threshold}")
            model[index[0][0],index[1][0],index[2][0],index[3][0]] += peak_value*self.settings['mgain']

            psf_shift = (index[2][0] + height // 2, index[3][0] + width // 2)
            residual = residual - peak_value*self.settings['mgain'] * np.roll(psf, psf_shift, axis=(1, 2))
            

            masked_residual=self.do_masking(residual,self.mask)
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
    
    
    def deconvolve_local(self,residual, model, psf,threshold):
        nchan, npol, height, width = residual.shape
        
        patch_half = 64

        # residual and model are numpy arrays with dimensions nchan x npol x height x width
        # psf is a numpy array with dimensions nchan x height x width

        # This file doesn't support multiple channels or polarizations:
        if nchan != 1 or npol != 1:
            raise NotImplementedError("nchan and npol must be one")
        
        
        if self.interactive:
            create_mask=self.mask_class(residual[0,0,:,:])
            plt.show()
            
            self.mask=np.expand_dims(create_mask.full_mask,axis=(0,1))
            self.interactive=create_mask.interactive
        elif not hasattr(self,'mask'):
            self.mask=np.ones(residual.shape,dtype=bool)
        elif self.mask.shape!=residual.shape:
            self.mask=np.expand_dims(self.mask,axis=(0,1))
            if self.mask.shape!=residual.shape:
                raise RuntimeError("Shape of provided mask does not match")
            
            
        

        masked_residual=self.do_masking(residual,self.mask)
        max_val=np.nanmax(masked_residual)
        index=np.where(np.abs(masked_residual-max_val)<1e-5)
        
        peak_value = residual[index[0][0],index[1][0],index[2][0],index[3][0]]

        mgain_threshold = abs(peak_value) * (1.0 - self.settings['mgain'])
        first_threshold = mgain_threshold
                       
        

        iteration_number=0
        while (abs(peak_value) > first_threshold and abs(peak_value)>threshold and iteration_number < self.settings['max_iterations']):
            print(f"peak={peak_value}, first threshold={first_threshold}")
            model[index[0][0],index[1][0],index[2][0],index[3][0]] += peak_value*self.settings['mgain']
            
            # Local Subtraction Coordinates
            # Peak location in the image
            py, px = index[2][0], index[3][0]
            # Center of the PSF (where the peak is located)
            cy, cx = height // 2, width // 2

            # Define bounds for the image and the PSF patch
            y_start, y_end = max(0, py - patch_half), min(height, py + patch_half + 1)
            x_start, x_end = max(0, px - patch_half), min(width, px + patch_half + 1)
            
            # Calculate the corresponding slices in the PSF array
            psf_y_start = cy - (py - y_start)
            psf_y_end   = cy + (y_end - py)
            psf_x_start = cx - (px - x_start)
            psf_x_end   = cx + (x_end - px)

            # 3. Apply subtraction only to the local window
            subtraction_term = peak_value * self.settings['mgain'] * psf[0, psf_y_start:psf_y_end, psf_x_start:psf_x_end]
            residual[0, 0, y_start:y_end, x_start:x_end] -= subtraction_term
            

            masked_residual=self.do_masking(residual,self.mask)
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
    
    def do_masking(self,data,mask):
        shape=data.shape
        data1=np.zeros_like(data)
        masked_data = np.where(self.mask, data, np.nan)
        return np.abs(masked_data)
    
    
        

    

            
    
    
    
    
    
            
    









    
    
     
    
