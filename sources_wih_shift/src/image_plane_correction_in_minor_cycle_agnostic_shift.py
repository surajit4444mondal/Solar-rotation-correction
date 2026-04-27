import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector, Button
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.path as mpath
import json
import tkinter as tk
from tkinter import filedialog


    
class image_plane_correction_minor_cycle():
    def __init__(self,forward_transform,backward_transform,msname,ref_time_isot,maskfile=None):
        self.forward_transform=forward_transform
        self.backward_transform=backward_transform
        self.msname=msname
        self.ref_time=Time(ref_time_isot,format='isot')    
        self.threshold=0.18
        self.max_major_cycle=3
        self.intervals=[[0,1],[1,2]]
        self.imagename='test_simulated_single_source_wsclean_self'
        self.final_image="test_self_major_minor"
        self.imsize=512
        self.cell=0.5  ###arcsec
        self.do_continue=False
        self.max_iterations=50
        self.mgain=0.1
        self.interactive=True
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
        
        result=self.deconvolve(residual, model, psf,self.threshold)
        
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

        mgain_threshold = abs(peak_value) * (1.0 - self.mgain)
        first_threshold = mgain_threshold
                        #max(meta.major_iter_threshold, meta.final_threshold, mgain_threshold)

        iteration_number=0
        while (abs(peak_value) > first_threshold and abs(peak_value)>threshold and iteration_number < self.max_iterations):
            print(f"peak={peak_value}, first threshold={first_threshold}")
            model[index[0][0],index[1][0],index[2][0],index[3][0]] += peak_value*self.mgain

            psf_shift = (index[2][0] + height // 2, index[3][0] + width // 2)
            residual = residual - peak_value*self.mgain * np.roll(psf, psf_shift, axis=(1, 2))
            

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
    
    
        

    
class MaskingSelector:
    def __init__(self, data):
        self.data = data
        self.full_mask = np.zeros(self.data.shape, dtype=bool)
        self.interactive=True
        self.fig, (self.ax_src, self.ax_mask) = plt.subplots(1, 2, figsize=(12, 6),sharex=True,sharey=True)
        self.ax_src.imshow(data, cmap='gray')
        self.ax_mask.set_title("Masked Result")
        plt.subplots_adjust(bottom=0.2)

        self.selections = []
        
        # Selectors
        self.rect = RectangleSelector(self.ax_src, self.on_select, interactive=False)
        self.circ = EllipseSelector(self.ax_src, self.on_select, interactive=False)
        self.circ.set_active(False)

        # Buttons
        ax_rect = plt.axes([0.1, 0.05, 0.08, 0.075])
        ax_circ = plt.axes([0.2, 0.05, 0.08, 0.075])
        ax_apply = plt.axes([0.3, 0.05, 0.08, 0.075])
        ax_clear = plt.axes([0.4, 0.05, 0.08, 0.075])

        self.btn_rect = Button(ax_rect, 'Rect')
        self.btn_circ = Button(ax_circ, 'Circle')
        self.btn_apply = Button(ax_apply, 'Apply Mask', color='lightgreen')
        self.btn_clear = Button(ax_clear, 'Clear')

        self.btn_rect.on_clicked(lambda x: self.toggle('r'))
        self.btn_circ.on_clicked(lambda x: self.toggle('c'))
        self.btn_apply.on_clicked(self.apply_mask)
        self.btn_clear.on_clicked(self.clear)
        self.mask_filename="mask.json"
        
        ax_save_mask = plt.axes([0.5, 0.05, 0.08, 0.075])
        ax_load_mask = plt.axes([0.6, 0.05, 0.08, 0.075])

        self.btn_save_mask = Button(ax_save_mask, 'Save Mask')
        self.btn_load_mask = Button(ax_load_mask, 'Load Mask')

        self.btn_save_mask.on_clicked(self.save_mask)
        self.btn_load_mask.on_clicked(self.load_mask)
        
        ax_save_patch = plt.axes([0.7, 0.05, 0.08, 0.075])
        ax_load_patch = plt.axes([0.8, 0.05, 0.08, 0.075])
        
        self.btn_save_patch = Button(ax_save_patch, 'Save Patches')
        self.btn_load_patch = Button(ax_load_patch, 'Load Patches')

        self.btn_save_patch.on_clicked(self.save_selections)
        self.btn_load_patch.on_clicked(self.load_selections)
        
        ax_noninteractive_patch = plt.axes([0.9, 0.05, 0.08, 0.075])
        self.btn_noninteractive_patch = Button(ax_noninteractive_patch, 'Non-interactive')
        self.btn_noninteractive_patch.on_clicked(self.go_noninteractive)
        
    def go_noninteractive(self, event=None):
        self.interactive=False

    def on_select(self, eclick, erelease):
        width = abs(erelease.xdata - eclick.xdata)
        height = abs(erelease.ydata - eclick.ydata)
        xmin, ymin = min(eclick.xdata, erelease.xdata), min(eclick.ydata, erelease.ydata)

        if self.rect.active:
            p = Rectangle((xmin, ymin), width, height, edgecolor='red', fill=False)
        else:
            center = (xmin + width/2, ymin + height/2)
            p = Ellipse(center, width, height, edgecolor='blue', fill=False)
        
        self.ax_src.add_patch(p)
        self.selections.append(p)
        self.fig.canvas.draw_idle()

    def toggle(self, mode):
        self.rect.set_active(mode == 'r')
        self.circ.set_active(mode == 'c')

    def apply_mask(self, event):
        # Create a coordinate grid for the image
        ny, nx = self.data.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T

        for patch in self.selections:
            # Get the path of the patch (works for both Rect and Ellipse)
            path = patch.get_path()
            patch_transform = patch.get_transform()
            combined_transform = patch_transform + self.ax_src.transData.inverted()
            data_path = path.transformed(combined_transform)
            grid_mask = data_path.contains_points(points).reshape((ny, nx))
            pos=np.where(grid_mask==True)
            self.full_mask |= grid_mask # Combine masks with OR

        # Mask the data: keep original where mask is True, else 0 (or NaN)
        masked_data = np.where(self.full_mask, self.data, np.nan)
        
        self.ax_mask.imshow(masked_data, cmap='gray')
        self.fig.canvas.draw_idle()

    def clear(self, event):
        for p in self.selections: p.remove()
        self.selections = []
        self.ax_mask.cla()
        self.fig.canvas.draw_idle()
        self.full_mask = np.zeros(self.data.shape, dtype=bool)
    
    def load_selections(self, event=None):
        # 1. Hide the tiny tkinter main window that pops up
        root = tk.Tk()
        root.withdraw() 
        
        # 2. Open the file browser
        file_path = filedialog.askopenfilename(
            title="Select Selection File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        # 3. Destroy root so it doesn't hang in the background
        root.destroy()

        if not file_path:
            return # User cancelled

        try:
            with open(file_path, 'r') as f:
                saved_data = json.load(f)
                
            # Clear current selections before loading new ones (optional)
            self.clear(None) 
            
            print (saved_data)
            for item in saved_data:
                if item['type'] == 'rectangle':
                    p = Rectangle((item['x'], item['y']), item['width'], item['height'], 
                                  edgecolor='red', fill=False, linewidth=2)
                elif item['type'] == 'ellipse':
                    p = Ellipse(item['center'], item['width'], item['height'], 
                                edgecolor='blue', fill=False, linewidth=2)
                
                self.ax_src.add_patch(p)
                self.selections.append(p)
                
                
            self.fig.canvas.draw_idle()
            print(f"Loaded {len(saved_data)} selections from {file_path}")
        except Exception as e:
            print(f"Error loading file: {e}")

    def save_selections(self, event=None):
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Selections As"
        )
        
        root.destroy()
        
        if file_path:
            # (Use the same dictionary logic from the previous step)
            data_to_save = []
            for patch in self.selections:
                shape_info = {}
                if isinstance(patch, Rectangle):
                    shape_info['type'] = 'rectangle'
                    shape_info['x'] = patch.get_x()
                    shape_info['y'] = patch.get_y()
                    shape_info['width'] = patch.get_width()
                    shape_info['height'] = patch.get_height()
                elif isinstance(patch, Ellipse):
                    shape_info['type'] = 'ellipse'
                    shape_info['center'] = patch.center # (x, y)
                    shape_info['width'] = patch.width
                    shape_info['height'] = patch.height
                    
                data_to_save.append(shape_info)
            
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"Saved to {file_path}")
            
    def load_mask(self, event=None):
        # 1. Hide the tiny tkinter main window that pops up
        root = tk.Tk()
        root.withdraw() 
        
        # 2. Open the file browser
        file_path = filedialog.askopenfilename(
            title="Select Selection File",
            filetypes=[("npy mask files", "*.npy"), ("All files", "*.*")]
        )
        
        # 3. Destroy root so it doesn't hang in the background
        root.destroy()

        if not file_path:
            return # User cancelled

        try:
            
            self.full_mask = np.load(file_path)
                
            print(f"Loaded mask from {file_path}")
        except Exception as e:
            print(f"Error loading file: {e}")
            
    def save_mask(self, event=None):
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            filetypes=[("npy mask files", "*.npy")],
            title="Save Selections As"
        )
        
        root.destroy()
        
        if file_path:
            np.save(file_path,self.full_mask)
        else:
            print(f"Error loading file: {file_path}")    
            
    
    
    
    
    
            
    









    
    
     
    
