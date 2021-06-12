import cv2
from matplotlib import pyplot as plt


def show_img(img):
    # cv2.imshow("test image", img)
    # cv2.waitKey(0)
    plt.figure()
    plt.imshow(img)
    plt.show()  # display it