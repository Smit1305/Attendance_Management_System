"""
Image Processing Lab Module for Semester VII Computer Vision
Implements 7 essential digital image processing operations with OpenCV:
1. Grayscale Conversion
2. Brightness & Contrast Adjustment
3. Histogram Computation & Histogram Equalization
4. Gaussian & Median Smoothing
5. Canny Edge Detection
6. Morphological Operations (Erosion, Dilation, Opening, Closing)
7. Image Thresholding (Binary, Inverted, Truncate, Tozero, Otsu)
"""

from typing import Dict, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """Validates that input array is a non-empty, 2D or 3D uint8 numpy image."""
    if image is None:
        return False, "Image is None."
    if not isinstance(image, np.ndarray):
        return False, f"Expected numpy.ndarray, got {type(image)}."
    if image.size == 0:
        return False, "Image array is empty."
    if image.ndim not in (2, 3):
        return False, f"Invalid dimensions: expected 2 or 3 dims, got {image.ndim}."
    return True, "Valid image."


# ============================================================================
# 1. Grayscale Conversion
# ============================================================================
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a color BGR image to single-channel Grayscale.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    OpenCV Function: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================================
# 2. Brightness & Contrast Adjustment
# ============================================================================
def adjust_brightness_contrast(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: int = 0,
) -> np.ndarray:
    """
    Adjusts contrast (alpha) and brightness (beta).
    Formula: g(x, y) = alpha * f(x, y) + beta
    - alpha > 1.0 increases contrast, alpha < 1.0 decreases contrast.
    - beta > 0 increases brightness, beta < 0 darkens image.
    OpenCV Function: cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ============================================================================
# 3. Histogram Display & Histogram Equalization
# ============================================================================
def equalize_histogram(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Performs Histogram Equalization to enhance global contrast.
    - For grayscale: applies cv2.equalizeHist directly.
    - For color: converts BGR to YCrCb, equalizes Y (Luminance) channel, converts back to BGR.
    Returns: (equalized_image, is_color)
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image if image.ndim == 2 else image[:, :, 0]
        equalized = cv2.equalizeHist(gray)
        return equalized, False
    else:
        # Convert BGR to YCrCb color space to decouple intensity from chroma
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return equalized, True


def plot_histogram_comparison(
    original: np.ndarray,
    equalized: np.ndarray,
) -> plt.Figure:
    """
    Generates a side-by-side Matplotlib histogram comparing
    intensity distribution before and after histogram equalization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    # Original Histogram
    if original.ndim == 2:
        hist_orig = cv2.calcHist([original], [0], None, [256], [0, 256])
        ax1.plot(hist_orig, color="black", lw=1.5)
        ax1.fill_between(range(256), hist_orig.flatten(), color="gray", alpha=0.3)
    else:
        colors = ("b", "g", "r")
        for i, col in enumerate(colors):
            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
            ax1.plot(hist, color=col, alpha=0.8, label=f"Channel {col.upper()}")
        ax1.legend(loc="upper right", fontsize=8)

    ax1.set_title("Original Intensity Histogram", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Pixel Value (0-255)")
    ax1.set_ylabel("Pixel Frequency")
    ax1.set_xlim([0, 256])
    ax1.grid(True, alpha=0.25)

    # Equalized Histogram
    if equalized.ndim == 2:
        hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
        ax2.plot(hist_eq, color="black", lw=1.5)
        ax2.fill_between(range(256), hist_eq.flatten(), color="gray", alpha=0.3)
    else:
        ycrcb = cv2.cvtColor(equalized, cv2.COLOR_BGR2YCrCb)
        hist_y = cv2.calcHist([ycrcb], [0], None, [256], [0, 256])
        ax2.plot(hist_y, color="purple", lw=1.5, label="Equalized Luminance (Y)")
        ax2.fill_between(range(256), hist_y.flatten(), color="purple", alpha=0.2)
        ax2.legend(loc="upper right", fontsize=8)

    ax2.set_title("Equalized Intensity Histogram", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Pixel Value (0-255)")
    ax2.set_xlim([0, 256])
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


# ============================================================================
# 4. Smoothing / Blurring Filters
# ============================================================================
def apply_smoothing(
    image: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Applies spatial neighborhood smoothing to reduce high-frequency noise.
    - 'gaussian': 2D Gaussian weighted kernel cv2.GaussianBlur(img, (k, k), sigma)
    - 'median': Non-linear median filter cv2.medianBlur(img, k) - best for salt & pepper noise.
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    # Kernel size must be positive and odd
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if method.lower() == "median":
        return cv2.medianBlur(image, kernel_size)
    else:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


# ============================================================================
# 5. Canny Edge Detection
# ============================================================================
def apply_canny(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """
    Applies Canny's 4-stage edge detection algorithm:
    1. Gaussian smoothing to remove noise.
    2. Sobel intensity gradient computation.
    3. Non-maximum suppression (thinning edges to 1 pixel).
    4. Hysteresis thresholding with low and high cutoffs.
    OpenCV Function: cv2.Canny(image, low, high, apertureSize, L2gradient)
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    gray = to_grayscale(image)
    edges = cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges


# ============================================================================
# 6. Morphological Operations
# ============================================================================
def apply_morphology(
    image: np.ndarray,
    operation: str = "dilation",
    kernel_shape: str = "rect",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """
    Applies non-linear neighborhood morphological transformations:
    - Erosion: Shrinks foreground objects, eliminates small speckles.
    - Dilation: Expands foreground objects, bridges small gaps.
    - Opening: Erosion followed by Dilation (smooths contours, removes noise).
    - Closing: Dilation followed by Erosion (fills small black holes and gaps).
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    shape_map = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    k_shape = shape_map.get(kernel_shape.lower(), cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(k_shape, (kernel_size, kernel_size))

    op_map = {
        "erosion": cv2.MORPH_ERODE,
        "dilation": cv2.MORPH_DILATE,
        "opening": cv2.MORPH_OPEN,
        "closing": cv2.MORPH_CLOSE,
    }

    selected_op = op_map.get(operation.lower(), cv2.MORPH_DILATE)
    return cv2.morphologyEx(image, selected_op, kernel, iterations=iterations)


# ============================================================================
# 7. Basic Image Thresholding
# ============================================================================
def apply_thresholding(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: Union[int, float, str] = 255,
    threshold_type: str = "binary",
) -> Tuple[np.ndarray, float]:
    """
    Segments an image into foreground and background by thresholding.
    Types:
    - 'binary': Pixels > thresh become max_val, else 0.
    - 'binary_inv': Pixels > thresh become 0, else max_val.
    - 'trunc': Pixels > thresh become thresh, else unchanged.
    - 'tozero': Pixels < thresh become 0, else unchanged.
    - 'otsu': Automatic global optimal threshold calculation via variance minimization.
    Returns: (thresholded_image, computed_threshold_value)
    """
    ok, err = validate_image(image)
    if not ok:
        raise ValueError(err)

    # Support flexible parameter ordering if threshold_type was passed as 3rd positional argument
    if isinstance(max_value, str):
        threshold_type = max_value
        max_val_num = 255
    else:
        max_val_num = float(max_value)

    gray = to_grayscale(image)

    type_map = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    }

    selected_type = type_map.get(threshold_type.lower(), cv2.THRESH_BINARY)
    actual_thresh, thresh_img = cv2.threshold(gray, float(threshold_value), max_val_num, selected_type)
    return thresh_img, actual_thresh
