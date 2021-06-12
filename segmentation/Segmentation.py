from segmentation.Seperator import seperator
from segmentation.Watershed import watershed
import cv2
from feature_extraction.GLCM_features import *
  
# %matplotlib inline 
class Segmentation:
    def __init__(self):
        super().__init__()

    def segmentation(self,images,image_names,patient_name,output_dir):
        procesed_images = []
        from pathlib import Path
        Path(output_dir+patient_name).mkdir(parents=True, exist_ok=True)
        for i,image in enumerate(images):
            canny, filtered = seperator(image)
            createGLCMImage(filtered)
            image_name = image_names[i]
            cv2.imwrite(output_dir+patient_name+'/'+ image_name + '.jpg', canny)
            print(output_dir+patient_name+'/'+ image_name + '.jpg')
            cv2.imwrite(output_dir + patient_name + '/' + image_name + '.png', filtered)

        

    def debugDisplay(self,image):
        cv2.imshow('sample image dicom',image)
        cv2.waitKey()