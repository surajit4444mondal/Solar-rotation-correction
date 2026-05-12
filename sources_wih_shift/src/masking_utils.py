import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector, Button
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.path as mpath
import json
import tkinter as tk
from tkinter import filedialog



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
