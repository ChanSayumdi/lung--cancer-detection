import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from util.util import *

from skimage.filters import gaussian, threshold_otsu
from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some of the starting Code is taken from ArnavJain, since it's more readable then my own
def generate_markers(image):
     # show_img(image)
    #Creation of the internal Marker
    marker_internal = image < -400
    # show_img(marker_internal)
    marker_internal = segmentation.clear_border(marker_internal)
    show_img(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    # show_img(marker_internal_labels)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    show_img(marker_internal)
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    # show_img(external_a)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    # show_img(external_b)
    marker_external = external_b ^ external_a
    # show_img(marker_external)
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    # show_img(marker_watershed)
    
    return marker_internal, marker_external, marker_watershed

def watershed(image):
    #Show some example markers from the middle        
    print("watershed")
    test_patient_internal, test_patient_external, test_patient_watershed = generate_markers(image)
    print ("Internal Marker")
    plt.imshow(test_patient_internal, cmap='gray')
    plt.show()
    print ("External Marker")
    plt.imshow(test_patient_external, cmap='gray')
    plt.show()
    print ("Watershed Marker")
    plt.imshow(test_patient_watershed, cmap='gray')
    plt.show()