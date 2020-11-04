from skimage.filters import gabor,gabor_kernel
from skimage import data, io
from matplotlib import pyplot as plt 
# scikit image python

def gabor():
    image = data.coins()
    gk = gabor_kernel(frequency=0.2)
    plt.figure()        
    io.imshow(gk.real)  
    io.show()  

    # detecting edges in a coin image
    # filt_real, filt_imag = gabor(image, frequency=0.6)
    # plt.figure()            
    # io.imshow(filt_real)    
    # io.show()        

    # # less sensitivity to finer details with the lower frequency kernel
    # filt_real, filt_imag = gabor(image, frequency=0.1)
    # plt.figure()            
    # io.imshow(filt_real)    
    # io.show()               