"""
...

"""

import numpy as np
import globalHDR
import scipy.ndimage as ndimage

original = globalHDR.read_image()
# globalHDR.show(original)

grey = original.astype(float).sum(2) / (255 * 3)
# globalHDR.show(grey)

blurry = ndimage.uniform_filter(grey, size=11)
globalHDR.show(blurry)

details = np.zeros(original.shape)
print("original:", original.shape, "details:", details.shape, "blurry:", blurry.shape)
for i in range(0, details.ndim):
    details[:, :, i] = original[:, :, i] - blurry
limit = details.mean() * 2 # * 3 # 0.5 # details.mean() # Threshold for hva som "er" detaljer
details[details > limit] = 1
details[details <= limit] = 0
globalHDR.show(details)

blurry_edit = globalHDR.edit_globally(blurry)
new = np.zeros(details.shape)
for i in range(0, details.ndim):
    new[:, :, i] = blurry_edit + details[:, :, i]
globalHDR.show(new)


# -blurry = ndimage.binary_dilation(binary, structure=np.ones((2, 2))).astype(binary.dtype)
# -#-globalHDR.show(blurry)




