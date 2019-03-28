"""
...

"""

import numpy as np
import globalHDR
import scipy.ndimage as ndimage


def filter_linear_spatial(im, limit=.5):
    globalHDR.show(im)
    
    grey = im.astype(float).sum(2) / (255 * 3)
    blurry = np.zeros(im.shape)
    #blurry = ndimage.uniform_filter(grey, size=15)
    for i in range(0, im.ndim):
        blurry[:, :, i] = ndimage.uniform_filter(grey, size=15)
    globalHDR.show(blurry)
    
                #details = np.zeros(im.shape)
    #print("im:", im.shape, "details:", details.shape, "blurry:", blurry.shape)
    details = im - blurry
                #for i in range(0, details.ndim):
                #    details[:, :, i] = im[:, :, i] - blurry
                #    # details = im - blurry

    #print(np.percentile(details, 90))
    limit = limit * 0 + np.percentile(details, 90)  # Dette betyr at kun 10% = detaljer
                                                    #  * 2 # * 3 # 0.5 # details.mean() # Threshold for hva som "er" detaljer
    details[details > limit] = 1
    details[details <= limit] = 0
    globalHDR.show(details)
    
    blurry_edited = globalHDR.edit_globally(blurry)  # sqrt er default
    # blurry_edited = globalHDR.edit_luminance(blurry)
                ##filtered = np.zeros(details.shape)
    filtered = blurry_edited + details
                ##for i in range(0, details.ndim):
                ##    filtered[:, :, i] = blurry_edited + details[:, :, i]
    globalHDR.show(filtered)
    
    globalHDR.compare(details, filtered)


filter_linear_spatial(globalHDR.read_image())
