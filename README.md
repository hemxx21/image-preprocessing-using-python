Image Preprocessing
This project includes various image preprocessing techniques to prepare and enhance images for further analysis, 
feature extraction, or model training. Below is a detailed list of the preprocessing steps implemented:

1. Loading the Image
-> Images are read using libraries like cv2 (OpenCV) and converted into arrays for processing.
-> Images can be loaded in different formats (e.g., BGR, RGB) depending on the requirements.

2. Image Enhancement
  Applied techniques to improve image quality, such as:
      -> Contrast Adjustment: Enhanced image visibility using Contrast Limited Adaptive Histogram Equalization (CLAHE).
      -> Brightness Adjustment: Increased or decreased brightness to ensure consistent image lighting.
      -> Gamma Correction: Adjusted the gamma levels to refine pixel intensity distribution.

3. Feature Extraction
  Extracted essential features from the image for analysis or machine learning purposes.
      -> HOG Features: Histogram of Oriented Gradients was used to capture structural features.
      -> Edge Detection: Identified prominent edges for feature extraction.

4. Data Augmentation
  To increase the dataset's diversity and prevent overfitting, the following augmentations were applied:

      -> Flipping: Horizontal and vertical flipping of images.
      -> Rotation: Rotated images by various angles to capture all orientations.
      -> Resizing: Scaled images to specific dimensions.
      -> Noise Addition: Added Gaussian noise to simulate real-world conditions.

5. Decreasing Pixel Density
    Reduced image resolution (downsampling) to create lower-dimensional representations, which can be useful for
    computational efficiency or simulating degraded conditions.

6. Image Sharpening
    Applied a sharpening kernel to highlight edges and details in the image, enhancing clarity.

7. Blurring the Image
    Used Gaussian and box blur filters to:
        -> Reduce noise.
        -> Smooth out the image for specific tasks like background removal.

8. Color Space Conversions
    Converted images to different color spaces for specialized processing:
        -> BGR to RGB: Used for visualization.
        -> Grayscale: Simplified images by removing color information while retaining intensity.

9. Channel Separation
    Extracted and displayed individual color channels to analyze:
        -> Red Channel
        -> Green Channel
        -> Blue Channel

10. Displaying and Visualization
    -> Visualized processed images at various stages for qualitative analysis.
    -> Used matplotlib for displaying images and histograms.


Purpose and Benefits
  These preprocessing techniques are crucial for:
    -> Enhancing image quality.
    -> Reducing noise and improving data reliability.
    -> Generating diverse training datasets through augmentation.
    -> Extracting relevant features for machine learning models.
    -> Reducing computational complexity through pixel downscaling.

Requirements
  -> Python Libraries: OpenCV, NumPy, Matplotlib, SciPy, and scikit-image.
  -> Supported Formats: JPEG, PNG, BMP, and other common image formats.




