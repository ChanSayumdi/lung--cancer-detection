import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import cv2
from util.util import *
from skimage.segmentation import watershed
# from numba import jit, cuda



from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from segmentation.Watershed import generate_markers

# @jit(target ="cuda")
def seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    # show_img(sobel_gradient)
    
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    # show_img(watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    # show_img(outline)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # show_img(outline)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # show_img(lungfilter)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    # show_img(lungfilter)
    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))
    # show_img(segmented)
    
    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


def seperator(image):
    #Some Testcode:
    test_segmented, test_lungfilter, test_outline, test_watershed, test_sobel_gradient, test_marker_internal, test_marker_external, test_marker_watershed = seperate_lungs(image)

    size = 50
    if not size % 2:
        size += 1
    kernel = np.ones((size, size), np.float32) / (size * size)
    filtered = cv2.filter2D(test_segmented, -1, kernel)
    # show_img(filtered)

    # Gray level convertion
    filtered = test_segmented.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127 * np.ones(test_segmented.shape, np.uint8)
    filtered = cv2.normalize(filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    filtered = filtered.astype(np.uint8)
    # show_img(filtered)

    norm_image = cv2.normalize(test_segmented, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    canny = cv2.Canny(cv2.convertScaleAbs(norm_image), 50, 20)
    # show_img(canny)

    return canny, filtered
    # plt.subplot(3,3,1),plt.imshow(image,cmap = 'gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,2),plt.imshow(test_marker_internal,cmap = 'gray')
    # plt.title('Internal Marker'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,3),plt.imshow(test_marker_external,cmap = 'gray')
    # plt.title('External Marker'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,4),plt.imshow(test_marker_watershed,cmap = 'gray')
    # plt.title('Watershed Marker'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,5),plt.imshow(filtered,cmap = 'gray')
    # plt.title('Sobel Gradient'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,6),plt.imshow(canny,cmap = 'gray')
    # plt.title('Watershed Image'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,7),plt.imshow(test_outline,cmap = 'gray')
    # plt.title('Outline after reinclusion'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,8),plt.imshow(test_lungfilter,cmap = 'gray')
    # plt.title('Lungfilter after closing'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(3,3,9),plt.imshow(test_segmented,cmap = 'gray')
    # plt.title('Segmented Lung'), plt.xticks([]), plt.yticks([])
    #
    #
    # plt.show()


def debugDisplay(image):
    cv2.imshow('test sobel',image)
    cv2.waitKey()