
# from skimage import measure
# from skimage.color import rgb2gray
# from skimage.filters import sobel
# import matplotlib.pyplot as pyplot
# import Image
# import cv2 as cv
# from cv2.cv2 import imread
# from matplotlib import pyplot as plt

# #for original image
# IMAGE_PATH = "data/train/sample/contour.jpg"
# img = Image.open(IMAGE_PATH)
# img.show()

# img = imread(IMAGE_PATH)



# img_gray = rgb2gray(img)

# img_edges = sobel(img_gray)

# contours=measure.find_contours(img_edges,0.2)

# fig,ax = plt.subplots()
# ax.imshow(img_edges, interpolation='nearest', cmap=plt.cm.gray)

# for n, contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.show()