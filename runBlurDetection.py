import numpy as np
import cv2

def detect_blur(image):
    # Convert image to grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = 20

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    h, w = image.shape
    cy, cx = int(h/2), int(w/2)
    size = 60
    fft_shift[cy-size:cy+size, cx-size:cx+size] = 0

    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    magnitude = 20*np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return mean <= threshold