import cv2
from util.fast_glcm import *
import numpy as np
from util.util import *


def createGLCMImage(images):

    glcm_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (300,300))
        h,w = img.shape


        std = fast_glcm_std(img)
        ma = fast_glcm_max(img)
        ent = fast_glcm_entropy(img)
        mean = fast_glcm.fast_glcm_mean(img)
        std = fast_glcm.fast_glcm_std(img)
        cont = fast_glcm.fast_glcm_contrast(img)
        diss = fast_glcm.fast_glcm_dissimilarity(img)
        homo = fast_glcm.fast_glcm_homogeneity(img)
        asm, ene = fast_glcm.fast_glcm_ASM(img)
        ma = fast_glcm.fast_glcm_max(img)
        ent = fast_glcm.fast_glcm_entropy(img)


        needed_multi_channel_img = np.zeros((img.shape[0], img.shape[1], 3))

        needed_multi_channel_img[:, :, 0] = std
        needed_multi_channel_img[:, :, 1] = ma
        needed_multi_channel_img[:, :, 2] = ent

        glcm_images.append(needed_multi_channel_img.astype(int))

        show_img(std)
        show_img(ma)
        show_img(ent)
        plt.figure(figsize=(10, 4.5))
        fs = 15
        plt.subplot(2, 5, 1)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(img)
        plt.title('original', fontsize=fs)

        plt.subplot(2, 5, 2)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(mean)
        plt.title('mean', fontsize=fs)

        plt.subplot(2, 5, 3)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(std)
        plt.title('std', fontsize=fs)

        plt.subplot(2, 5, 4)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(needed_multi_channel_img)
        plt.title('contrast', fontsize=fs)

        plt.subplot(2, 5, 5)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(diss)
        plt.title('dissimilarity', fontsize=fs)

        plt.subplot(2, 5, 6)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(homo)
        plt.title('homogeneity', fontsize=fs)

        plt.subplot(2, 5, 7)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(asm)
        plt.title('ASM', fontsize=fs)

        plt.subplot(2, 5, 8)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(ene)
        plt.title('energy', fontsize=fs)

        plt.subplot(2, 5, 9)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(ma)
        plt.title('max', fontsize=fs)

        plt.subplot(2, 5, 10)
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.imshow(ent)
        plt.title('entropy', fontsize=fs)

        plt.tight_layout(pad=0.5)
        plt.savefig('img/output.jpg')
        plt.show()

    return np.array(glcm_images)
