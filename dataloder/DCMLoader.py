import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom as dicom
import os
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


class DCMLoader:
    def __init__(self):
        super().__init__()

    # Load the scans in given folder path
    def load_scan(self, path):
        all_filenames = []
        for dirpath, dirnames, filenames in os.walk(path):
            all_filenames += [os.path.join(dirpath, filename) for filename in filenames]
        all_filenames.sort()

        slices = []
        filenames_out = []

        for filename in all_filenames:
            try:
                img = dicom.read_file(filename)
                if 512 == img.pixel_array.shape[0] :
                    slices.append (img)
                    file = filename.split('\\')[-1]
                    file = file.split('.')[0]
                    filenames_out.append(file)
            except:
                print('empty-folder')
        s_slice = 0

        if len(slices) > 0:
            # slices.sort(key = lambda x: int(x.InstanceNumber))
            for i in range(len(slices)):
                try:
                    slice_thickness = np.abs(slices[0 + i].ImagePositionPatient[2] - slices[1 + i].ImagePositionPatient[2])
                    break
                except:
                    try:
                        slice_thickness = np.abs(slices[0 + i].SliceLocation - slices[1 + i].SliceLocation)
                        break
                    except:
                        s_slice += 1

            for s in slices:
                s.SliceThickness = slice_thickness
            
        return slices, filenames_out

    def get_pixels_hu(self, scans):
        image = np.stack([s.pixel_array for s in scans])
        # tem_images = np.stack([])
        # for s in scans:
        #     try:
        #         tem_images.

        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # Convert to Hounsfield units (HU)
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope
        
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)

    def load(self, dataPath , num):
        patients = os.listdir(dataPath)
        patients.sort()

        test_patient_scans , filenames_out = self.load_scan(dataPath + patients[num])
        if len(test_patient_scans) > 0 :
            test_patient_images = self.get_pixels_hu(test_patient_scans)
        else :
            test_patient_images = []
        return test_patient_images , filenames_out , patients[num]