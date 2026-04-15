import numpy as np
from astropy.io import fits

'''
The function below is adapted from Offringa's example given in
https://gitlab.com/aroffringa/wsclean/-/blob/master/scripts/python-examples/simple-deconvolution-example.py


This function does not work if it is called with intervals-out
'''


def deconvolve(residual, model, psf, meta):
    nchan, npol, height, width = residual.shape
    print("Python deconvolve() function was called for "+\
             f"{width} x {height} x {npol} (npol) x {nchan} (chan) dataset")

    # residual and model are numpy arrays with dimensions nchan x npol x height x width
    # psf is a numpy array with dimensions nchan x height x width

    # This file doesn't support multiple channels or polarizations:
    if nchan != 1 or npol != 1:
        raise NotImplementedError("nchan and npol must be one")
        
    print (meta)

    # meta contains several useful meta data:
    # meta.channels is an array, each element has properties 'frequency' and 'weight'
    
    # Furthermore, the meta class has properties major_iter_threshold, final_threshold,
    # mgain, iteration_number and max_iterations. These are demonstrated below.
    # iteration_number can be modified, and will keep its value when deconvolve()
    # is called again.

    # find the largest peak in residual
    peak_index = np.unravel_index(np.argmax(residual), residual.shape)
    peak_value = residual[peak_index]

    mgain_threshold = peak_value * (1.0 - meta.mgain)
    first_threshold = max(meta.major_iter_threshold, meta.final_threshold, mgain_threshold)
    ### Surajit: I am not using mgain threshold, otherwise every minor cycle will have only 1 subtraction

    
    
    while (peak_value > first_threshold and meta.iteration_number < meta.max_iterations):
        print(f"Starting iteration {meta.iteration_number}, peak={peak_value}, first threshold={first_threshold},major_iteration_threshold={meta.major_iter_threshold}")
        model[peak_index] += peak_value*meta.mgain

        psf_shift = (peak_index[2] + height // 2, peak_index[3] + width // 2)
        residual = residual - peak_value*meta.mgain * np.roll(psf, psf_shift, axis=(1, 2))
        

        peak_index = np.unravel_index(
            np.argmax(residual), residual.shape
        )
        peak_value = residual[peak_index]

        meta.iteration_number = meta.iteration_number + 1
        
        # find the largest peak
        peak_index = np.unravel_index(np.argmax(residual), residual.shape)
        peak_value = residual[peak_index]
        
        

    print(f"Stopped after iteration {meta.iteration_number}, peak={peak_value}")

    # Fill a dictionary with values that wsclean expects:
    result = dict()
    result["residual"] = residual
    result["model"] = model
    result["level"] = peak_value
    result["continue"] = False#(peak_value > meta.final_threshold and \
                            #meta.iteration_number < meta.max_iterations)

    print("Finished deconvolve()")
    return result
    
