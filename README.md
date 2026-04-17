# MiniCV Library - Milestone 1

## Project Description
MiniCV is a reusable Python image-processing library built from scratch using NumPy, Matplotlib, and the Python standard library.  
It implements a small and well-defined subset of OpenCV functionality, including filtering, thresholding, geometric transformations, feature extraction, drawing primitives, and canvas operations.

## Module Structure
- `io.py` → image reading, saving, RGB to grayscale, grayscale to RGB
- `utils.py` → validation, normalization, clipping, padding
- `filtering.py` → convolution, filters, thresholding, Sobel, bit-plane slicing, log transform
- `transforms.py` → resizing, translation, rotation
- `features.py` → histogram, histogram equalization, feature descriptors
- `drawing.py` → point, line, rectangle, polygon, text rendering

## Pipeline
1. Image reading and preprocessing
2. Filtering (Mean, Gaussian, Median)
3. Edge detection (Sobel)
4. Thresholding (Otsu, Adaptive)
5. Feature extraction
6. Geometric transformations
7. Drawing primitives and canvas operations

## Math & Algorithms Notes

### 1. True 2D Convolution
The core filtering operation is implemented as true 2D convolution by flipping the kernel before sliding it over the image:

\[
O(x,y) = \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} I(x+i, y+j) \cdot K_f(i,j)
\]

where \( K_f \) is the flipped kernel.

### 2. Gaussian Filter
Gaussian smoothing is used for noise reduction. The kernel is generated using:

\[
G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
\]

### 3. Sobel Edge Detection
The Sobel operator computes image gradients in horizontal and vertical directions:

\[
G = \sqrt{G_x^2 + G_y^2}
\]

### 4. Histogram Equalization
Histogram equalization improves contrast by redistributing intensity values using the cumulative distribution function (CDF).

### 5. Otsu Thresholding
Otsu’s method automatically selects the threshold that minimizes intra-class variance:

\[
\sigma_w^2(t) = \omega_0(t)\sigma_0^2(t) + \omega_1(t)\sigma_1^2(t)
\]

## Note
True 2D convolution is implemented by flipping the kernel before applying the sliding window operation.

## Results
Add your generated result image here after saving it from the program.

![Results](results.png)
