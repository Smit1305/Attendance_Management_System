"""
Unit tests for image_processing.py
Tests image validation, grayscale conversion, brightness/contrast,
histogram equalization, smoothing, Canny edge detection, morphology, and thresholding.
"""

import os
import unittest
import cv2
import numpy as np

import image_processing


class TestImageProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a reliable synthetic color image
        cls.test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Background gradient
        for y in range(100):
            for x in range(100):
                cls.test_img[y, x] = [x * 2, y * 2, 120]
        # Draw shapes for edges and morphology
        cv2.circle(cls.test_img, (50, 50), 20, (255, 255, 255), -1)
        cv2.rectangle(cls.test_img, (10, 10), (30, 30), (0, 255, 0), -1)

    def test_image_validation(self):
        """Tests input image validation logic."""
        ok, _ = image_processing.validate_image(self.test_img)
        self.assertTrue(ok)

        ok_none, _ = image_processing.validate_image(None)
        self.assertFalse(ok_none)

        ok_empty, _ = image_processing.validate_image(np.array([]))
        self.assertFalse(ok_empty)

    def test_to_grayscale(self):
        """Tests grayscale conversion and channel reduction."""
        gray = image_processing.to_grayscale(self.test_img)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (100, 100))

        # Idempotency: passing grayscale should return copy unchanged
        gray2 = image_processing.to_grayscale(gray)
        self.assertEqual(gray2.ndim, 2)

    def test_adjust_brightness_contrast(self):
        """Tests brightness and contrast linear adjustment."""
        adjusted = image_processing.adjust_brightness_contrast(self.test_img, alpha=1.5, beta=30)
        self.assertEqual(adjusted.shape, self.test_img.shape)
        # Higher alpha and positive beta should increase mean pixel value
        self.assertGreaterEqual(adjusted.mean(), self.test_img.mean())

    def test_histogram_equalization(self):
        """Tests color and grayscale histogram equalization."""
        eq_col, is_col = image_processing.equalize_histogram(self.test_img)
        self.assertTrue(is_col)
        self.assertEqual(eq_col.shape, self.test_img.shape)

        gray = image_processing.to_grayscale(self.test_img)
        eq_gray, is_col_gray = image_processing.equalize_histogram(gray)
        self.assertFalse(is_col_gray)
        self.assertEqual(eq_gray.shape, (100, 100))

    def test_apply_smoothing(self):
        """Tests Gaussian and Median filters."""
        g_smooth = image_processing.apply_smoothing(self.test_img, method="gaussian", kernel_size=5)
        self.assertEqual(g_smooth.shape, self.test_img.shape)

        m_smooth = image_processing.apply_smoothing(self.test_img, method="median", kernel_size=5)
        self.assertEqual(m_smooth.shape, self.test_img.shape)

    def test_canny_edge_detection(self):
        """Tests multi-stage Canny edge detection."""
        edges = image_processing.apply_canny(self.test_img, low_threshold=50, high_threshold=150)
        self.assertEqual(edges.ndim, 2)
        self.assertEqual(edges.shape, (100, 100))
        # Unique values in Canny output must be binary (0 and 255)
        unique_vals = set(np.unique(edges))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_morphology(self):
        """Tests erosion and dilation operations."""
        gray = image_processing.to_grayscale(self.test_img)
        dilated = image_processing.apply_morphology(gray, operation="dilation", kernel_size=5)
        eroded = image_processing.apply_morphology(gray, operation="erosion", kernel_size=5)
        self.assertEqual(dilated.shape, gray.shape)
        self.assertEqual(eroded.shape, gray.shape)
        # Dilation expands bright regions (mean increases); Erosion shrinks them (mean decreases)
        self.assertGreaterEqual(dilated.mean(), eroded.mean())

    def test_thresholding(self):
        """Tests binary and Otsu's automated thresholding."""
        thresh_bin, val_bin = image_processing.apply_thresholding(
            self.test_img, threshold_value=127, threshold_type="binary"
        )
        self.assertEqual(thresh_bin.ndim, 2)
        unique_vals = set(np.unique(thresh_bin))
        self.assertTrue(unique_vals.issubset({0, 255}))

        thresh_otsu, val_otsu = image_processing.apply_thresholding(
            self.test_img, threshold_type="otsu"
        )
        self.assertGreater(val_otsu, 0.0)
        self.assertEqual(thresh_otsu.shape, (100, 100))


if __name__ == "__main__":
    unittest.main()
