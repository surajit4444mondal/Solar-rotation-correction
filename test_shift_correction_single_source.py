import numpy as np
#from casatools import image,ms
#from casatasks import ft,uvsub,delmod,split
import os

def transform_image(imagename,x1=0,x2=512,y1=0,y2=512,shiftx=0,shifty=0):
	ia.open(imagename)
	try:
		imagedata=ia.getchunk()
		shape=imagedata.shape
		print (shape)
		data=np.zeros_like(imagedata)
		for i in range(x1,x2):
			for j in range(y1,y2):
				try:
					data[i,j,0,0]=imagedata[i+shiftx,j+shifty,0,0]
				except IndexError:
					pass
		ia.putchunk(data)
	finally:
		ia.close()
	return

imagename='vla_c_simulated_single_point_shifted_image.model'
outfile1='vla_c_simulated_single_point_shift_corrected_s1.model'
msname='vla_c_simulated_single_point_shifted.ms'
outms='vla_c_simulated_single_point_shift_corrected.ms'

combined_vis='vla_c_simulated_double_point_shift_corrected_combined.ms'

os.system("rm -rf temp.ms")
os.system("cp -r "+msname+" temp.ms")

ft(vis='temp.ms',model=imagename,usescratch=True)
uvsub(vis='temp.ms')  #### now corrected data is the purely the residual
delmod(vis='temp.ms',scr=True)

os.system("cp -r "+imagename+" "+outfile1)
transform_image(outfile1,x1=223,x2=309,y1=215,y2=294,shiftx=15)
ft(vis='temp.ms',model=outfile1,usescratch=True)  ### model now has the visibilities of the shifted data


uvsub(vis='temp.ms',reverse=True)   #### Now the corrected data column will add the residuals +
				     #### visibilities correspodning to the shifted image

if os.path.isdir(outms)==True:
	os.system("rm -rf "+outms)

if os.path.isdir(combined_vis)==True:
	os.system("rm -rf "+combined_vis)
	
split(vis='temp.ms',outputvis=outms,datacolumn='corrected')

concat(vis=[outms,'vla_c_simulated_single_point.ms'],concatvis=combined_vis)

tclean(vis=combined_vis,imagename='combined_image_temp',imsize=[512],cell='0.5arcsec',niter=800,interactive=True)

model="combined_image_temp.model"
os.system("rm -rf "+outfile1)
os.system("cp -r "+model+" "+outfile1)
transform_image(outfile1,x1=223,x2=309,y1=215,y2=294,shiftx=-15)

tclean(vis=msname,imagename='vla_c_simulated_single_point_shifted_image_second_round',\
		imsize=[512],cell='0.5arcsec',niter=800,interactive=True,startmodel=outfile1)

imagename='vla_c_simulated_single_point_shifted_image_second_round.model'

os.system("rm -rf temp.ms")
os.system("cp -r "+msname+" temp.ms")

ft(vis='temp.ms',model=imagename,usescratch=True)
uvsub(vis='temp.ms')  #### now corrected data is the purely the residual
delmod(vis='temp.ms',scr=True)


os.system("rm -rf "+outfile1)
os.system("cp -r "+imagename+" "+outfile1)

transform_image(outfile1,x1=223,x2=309,y1=215,y2=294,shiftx=15)
ft(vis='temp.ms',model=outfile1,usescratch=True)  ### model now has the visibilities of the shifted data

uvsub(vis='temp.ms',reverse=True)   #### Now the corrected data column will add the residuals +
				     #### visibilities correspodning to the shifted image

if os.path.isdir(outms)==True:
	os.system("rm -rf "+outms)
	
if os.path.isdir(combined_vis)==True:
	os.system("rm -rf "+combined_vis)
	
split(vis='temp.ms',outputvis=outms,datacolumn='corrected')

concat(vis=[outms,'vla_c_simulated_single_point.ms'],concatvis=combined_vis)

tclean(vis=combined_vis,imagename='combined_image_temp_second_round',imsize=[512],cell='0.5arcsec',niter=800,interactive=True)

