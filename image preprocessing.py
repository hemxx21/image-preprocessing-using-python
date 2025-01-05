#!/usr/bin/env python
# coding: utf-8

# # installing required packages

# In[1]:


get_ipython().system('pip install opencv-python Pillow matplotlib numpy')


# # Importing necessary libraries

# In[2]:


import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# # Displaying original image

# In[3]:


# Load the image using OpenCV
image_path = 'MainAfter.jpg'
image = cv2.imread(image_path)

# Display the image using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()


# # Difference between CV2 image and matplotlib image

# In[16]:


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV2 Image')
axs[1].set_title('Matplotlib Image')
plt.show()


# # displaying the image dimensions

# In[4]:


img_mpl = plt.imread('MainAfter.jpg')


# In[5]:


image.shape


# # resizing image

# In[6]:


# Resize to 224x224
resized_image = cv2.resize(image, (224, 224))
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('Resized Image')
plt.axis('off')
plt.show()


# In[36]:


#size of the resized image

resized_image.shape 


# # Normalising the image

# In[7]:


# Normalize to [0, 1]
normalized_image = resized_image / 255.0
print(f"Normalized Image Shape: {normalized_image.shape}")


# # Data Augmentation steps

# In[9]:


# Flip horizontally
flipped_image = cv2.flip(resized_image, 1)
plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
plt.title('Flipped Image')
plt.axis('off')
plt.show()


# In[10]:


# Rotate by 30 degrees
(h, w) = resized_image.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated_image = cv2.warpAffine(resized_image, matrix, (w, h))
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')
plt.axis('off')
plt.show()


# # Adding noise to the image

# In[11]:


# Add Gaussian noise
noise = np.random.normal(0, 0.1, resized_image.shape)
noisy_image = np.clip(resized_image / 255.0 + noise, 0, 1)
plt.imshow(noisy_image)
plt.title('Noisy Image')
plt.axis('off')
plt.show()


# In[12]:


# Adding batch dimension: Main step for image processing in neural networks and deep learning

input_image = np.expand_dims(normalized_image, axis=0)
print(f"Input Image Shape: {input_image.shape}")


# In[13]:


#To analyse the intensity values using histogram

pd.Series(image.flatten()).plot(kind='hist',
                                  bins=50,
                                  title='Distribution of Pixel Values')
plt.show()


# In[15]:


# Display RGB Channels of our image

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image[:,:,0], cmap='Reds')
axs[1].imshow(image[:,:,1], cmap='Greens')
axs[2].imshow(image[:,:,2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[0].set_title('Red channel')
axs[1].set_title('Green channel')
axs[2].set_title('Blue channel')
plt.show()


# In[19]:


# Sharpen Image

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])

sharpened = cv2.filter2D(image, -1, kernel_sharpening)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(sharpened)
ax.axis('off')
ax.set_title('Sharpened Image')
plt.show()


# In[20]:


# Blurring the image

kernel_3x3 = np.ones((100, 100), np.float32) / 10000
blurred = cv2.filter2D(img_mpl, -1, kernel_3x3)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(blurred)
ax.axis('off')
ax.set_title('Blurred Image')
plt.show()


# In[8]:


# Convert to grayscale

gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()


# # Feature Extraction

# In[38]:


import cv2
import numpy as np
from skimage.feature import hog
from matplotlib import pyplot as plt

# Load the image
image_path = 'MainAfter.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to a standard size
image_resized = cv2.resize(image, (128, 128))

# 1. Histogram of Oriented Gradients (HOG) Features
def extract_hog_features(image):
    features, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True,
    )
    return features, hog_image

hog_features, hog_image = extract_hog_features(image_resized)



color_histogram = extract_color_histogram(image_resized)

# 3. Edge Detection Features (Canny)
def extract_edge_features(image):
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    return edges

edges = extract_edge_features(image_resized)

# Display the results
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("HOG Features")
plt.imshow(hog_image, cmap='gray')
plt.axis('off')


plt.subplot(2, 2, 4)
plt.title("Edge Detection")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Combine features into a single feature vector (example for a machine learning model)
combined_features = np.hstack([hog_features, color_histogram])
print(f"Combined Features Shape: {combined_features.shape}")


# # Image Enhancement

# In[40]:


import cv2
import numpy as np
from skimage import exposure
from scipy import ndimage

def enhance_image(image):
    """
    Comprehensive image enhancement including contrast, brightness, and sharpness.
    """
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Brightness adjustment
    brightness_factor = 1.2
    image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=10)
    
    # Gamma correction
    gamma = 1.2
    image = exposure.adjust_gamma(image, gamma)
    
    # Sharpen using unsharp mask
    blurred = ndimage.gaussian_filter(image, sigma=1.0)
    image = image + 0.6 * (image - blurred)
    
    # Ensure image is in uint8 format before color saturation
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Color saturation enhancement
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Final normalization
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Read and enhance image
img = cv2.imread('MainAfter.jpg')
if img is None:
    print("Error: Image not found. Check the file path.")
else:
    enhanced = enhance_image(img)
    cv2.imwrite('enhanced_img.jpg', enhanced)
    print("Enhanced image saved as 'enhanced_img.jpg'")


# In[34]:


#decreases pixel

# Load the image
import cv2
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('MainAfter.jpg')

# Check if the image is loaded properly
if image is None:
    print("Error: Could not read the image. Please check the file path.")
else:
    # Resize the image
    img_resized = cv2.resize(image, None, fx=0.1, fy=0.1)

    # Convert BGR to RGB for correct color display
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Display the resized image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_resized_rgb)
    ax.axis('off')
    plt.show()


# In[ ]:




