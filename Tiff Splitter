import numpy as np
from skimage.external.tifffile import imread, TiffFile
import os
from skimage import io

# paths
original_file_location = r"/media/matthew/Seagate Expansion Drive/Zebrafish/060819_GCAMP_Tectum/Fish_3/"
fish_name = "060819_GCAMP_Tectum_F3_00002.tif"
new_folder_location = original_file_location + "Split_Tiff"
index_to_discard = 6

def tiff_split():  # define function tiff_split, and with two inputs

    os.mkdir(new_folder_location, 0o755)

    img = imread(original_file_location + fish_name)        # Load Tiff Into Array
    counter = 1                                 # Variable to Count Through All Original Tiffs
    names = 1                                   # Variable to Count Through Tiffs We Are Keeping
    
    for i in range(img.shape[0]):               # i loop through number of frames in each tif file
        if counter % index_to_discard == 0:     # if counter is divisible by index to discard, do not save
            counter += 1                        # counter iterate when it is at 6
        else:

            io.imsave(arr=img[i, :, :], fname=new_folder_location + "/" + str(names).zfill(6) + ".tif")
            counter += 1  # else if not divisible by 6,all other values save the ith image in array and name
            names += 1  # iterate counter and also name, but not name if i=6 so we get consecutive named files

tiff_split()
