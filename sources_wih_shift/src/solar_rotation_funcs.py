import numpy as np
from astropy.io import fits
import os
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
from image_plane_correction_in_minor_cycle_agnostic_shift import image_plane_correction_minor_cycle as img_corr
from astropy.coordinates import EarthLocation, SkyCoord
import sunpy.map as smap
from sunpy.coordinates import frames, sun,propagate_with_solar_surface
from astropy import units as u
from astropy.coordinates import ICRS
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from astropy.wcs import WCS
from reproject import reproject_interp
from astroquery.jplhorizons import Horizons

def get_observatory_coord(observatory):
    if observatory == 'EVLA' or observatory == 'VLA' or observatory=='-5':
        site_coord = '252.382,34.0788132,2.11447'
    elif observatory == 'EOVSA' or observatory == 'FASR':
        site_coord = '241.713,37.2332,1.20713'
    elif observatory == 'OVRO_MMA' or observatory == "OVRO-LWA" or observatory == 'OVRO':
        site_coord = '241.718406,37.240115,1.18835'
    elif observatory == 'ALMA' or observatory == '-7':
        site_coord = '292.2452521,-23.029211,5.07489'
    elif observatory == 'GMRT' or observatory == 'uGMRT':
        site_coord = '74.050508,19.093096,.60663'
    elif observatory == 'geocentric' or observatory == '500':
        site_coord = '0.0,0.0,-6378.137'
    else:
        raise RuntimeError('Observatory {} not recognized.'.format(observatory))
        
    return site_coord

def convert_fits_to_map(imagename,observatory, use_phacenter):
    with fits.open(imagename) as hdul:
        header = hdul[0].header
        data = np.squeeze(hdul[0].data)
    obstime = Time(header['date-obs'])
    frequency = header['crval3']*u.Hz
    
    observatory_coord_str=get_observatory_coord(observatory)
    coords=observatory_coord_str.split(',')

    observatory_loc = EarthLocation(lat=float(coords[1])*u.deg, lon=float(coords[0])*u.deg,height=float(coords[2])*1e3*u.m)
    observatory_coord = SkyCoord(observatory_loc.get_itrs(Time(obstime)))
        
    if use_phacenter:
        reference_coord = SkyCoord(header['crval1']*u.deg, header['crval2']*u.deg,
                               frame='icrs',
                               obstime=obstime,
                               distance=sun.earth_distance(obstime),
                               equinox='J2000')
        
        
        
        reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=observatory_coord))
    else:
        reference_coord_arcsec=SkyCoord(0*u.arcsec,0*u.arcsec,frame=frames.Helioprojective(observer=observatory_coord))
    
    cdelt1 = (np.abs(header['cdelt1'])*u.deg).to(u.arcsec)
    cdelt2 = (np.abs(header['cdelt2'])*u.deg).to(u.arcsec)
    
    P1 = sun.P(obstime)
    
    new_header = smap.make_fitswcs_header(data, reference_coord_arcsec,
                                           reference_pixel=u.Quantity([header['crpix1']-1, header['crpix2']-1]*u.pixel),
                                           scale=u.Quantity([cdelt1, cdelt2]*u.arcsec/u.pix),
                                           rotation_angle=-P1,  ### note the -P1 sign
                                           wavelength=frequency.to(u.MHz),
                                           observatory=observatory)
    radio_map = smap.Map(data, new_header)
    
    return radio_map
    
def convert_model_data_to_map(model_data,header,ref_time,observatory,use_phacenter):
    obstime=Time(ref_time,format='isot')
    frequency = header['crval3']*u.Hz
    
    observatory_coord_str=get_observatory_coord(observatory)
    coords=observatory_coord_str.split(',')

    observatory_loc = EarthLocation(lat=float(coords[1])*u.deg, lon=float(coords[0])*u.deg,height=float(coords[2])*1e3*u.m)
    observatory_coord = SkyCoord(observatory_loc.get_itrs(Time(obstime)))
    
    if use_phacenter:
        reference_coord = SkyCoord(header['crval1']*u.deg, header['crval2']*u.deg,
                               frame='icrs',
                               obstime=obstime,
                               distance=sun.earth_distance(obstime),
                               equinox='J2000')
        
        reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=observatory_coord))
    else:
        reference_coord_arcsec=SkyCoord(0*u.arcsec,0*u.arcsec,frame=frames.Helioprojective(observer=observatory_coord))
    
    cdelt1 = (np.abs(header['cdelt1'])*u.deg).to(u.arcsec)
    cdelt2 = (np.abs(header['cdelt2'])*u.deg).to(u.arcsec)
    
    P1 = sun.P(obstime)
    
    new_header = smap.make_fitswcs_header(model_data, reference_coord_arcsec,
                                           reference_pixel=u.Quantity([header['crpix1']-1, header['crpix2']-1]*u.pixel),
                                           scale=u.Quantity([cdelt1, cdelt2]*u.arcsec/u.pix),
                                           rotation_angle=-P1,  ### note the -P1 sign
                                           wavelength=frequency.to(u.MHz),
                                           observatory=observatory)
    radio_map = smap.Map(model_data, new_header)
    return radio_map
        

def forward_transform_image(data,imagename,ref_time,observatory,use_phacenter):
    header=fits.getheader(imagename)
    gmrt_map=convert_model_data_to_map(data,header,ref_time,observatory,use_phacenter=use_phacenter)
    gmrt_meta=gmrt_map.meta

    
    timestamp_isot=header['DATE-OBS']
    time_obj=Time(timestamp_isot,format='isot')
    
    model_data=gmrt_map.data
    
    model_data_rot=np.zeros_like(model_data)

    pos=np.where(abs(model_data)>1e-3)
    
    naxis=gmrt_meta['naxis1']
    
    xrot=np.zeros(int(naxis*naxis))
    yrot=np.zeros_like(xrot)

    i=0
    for y1,x1 in zip(pos[0],pos[1]):
        coord=gmrt_map.pixel_to_world(x1*u.pix,y1*u.pix)
        if coordinate_is_on_solar_disk(coord)==True:
            rotated_coord = solar_rotate_coordinate(coord, time=time_obj)
            coord_new=gmrt_map.world_to_pixel(rotated_coord)
            xrot[i]=coord_new.x.value
            yrot[i]=coord_new.y.value
	      
        else:
            xrot[i]=x1
            yrot[i]=y1
        
        model_data_rot[int(yrot[i]),int(xrot[i])]=model_data[y1,x1]
        i+=1

    pos=np.where(np.isnan(model_data_rot)==True)
    model_data_rot[pos]=0
    
    return model_data_rot

def backward_transform_image(imagename,ref_time,observatory,use_phacenter):
    gmrt_map=convert_fits_to_map(imagename,observatory,use_phacenter=use_phacenter)
    gmrt_meta=gmrt_map.meta
    

    coord_frame=gmrt_map.coordinate_frame
    outtime=ref_time
    out_frame=frames.Helioprojective(observer=coord_frame.observer,obstime=outtime,rsun=coord_frame.rsun)

    P1 = sun.P(outtime)
    
    if not use_phacenter:
        outtime=Time(outtime)
        
        obj = Horizons(id='10', location='500', 
                   epochs={'start': outtime.iso, 'stop': (outtime+TimeDelta(60,format='sec')).iso, 'step': '1h'})
       
        eph = obj.ephemerides()
        ra=eph['RA'].value
        dec=eph['DEC'].value
        gmrt_meta['crval1']=ra[0]
        gmrt_meta['crval2']=dec[0]

    out_center = SkyCoord(gmrt_meta['CRVAL1']*u.arcsec, gmrt_meta['CRVAL2']*u.arcsec, frame=out_frame)
    header = smap.make_fitswcs_header(gmrt_map.data.shape,
                                           out_center,
                                           scale=u.Quantity(gmrt_map.scale),
                                           rotation_angle=-P1)
                                           

    out_wcs = WCS(header)
    
    

    with propagate_with_solar_surface():
        gmrt_map_difrot = gmrt_map.reproject_to(out_wcs)
        
    rotated_data=gmrt_map_difrot.data
    
    #fig=plt.figure()
    #ax=fig.add_subplot(121,projection=gmrt_map.wcs)
    #gmrt_map.plot(axes=ax)
    #ax=fig.add_subplot(122,projection=gmrt_map_difrot.wcs)#,sharex=ax,sharey=ax)
    #gmrt_map_difrot.plot(axes=ax)
    #plt.show()

        
    hpc_coords = all_coordinates_from_map(gmrt_map_difrot)

    
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / gmrt_map_difrot.rsun_obs

    # Create mask: True for off-disk (r > 1)
    off_disk_mask = r > 1

    ### If outside disc, no rotation correction is done.
    rotated_data[off_disk_mask] = gmrt_map.data[off_disk_mask]

    return np.expand_dims(rotated_data,axis=(0,1))
