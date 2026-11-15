"""Preprocessing and GLCM/HSV feature extraction utilities for corn kernels."""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURE_PATCH_SIZE = (128, 128)
FEATURE_NAMES = [
    "contrast",
    "correlation",
    "energy",
    "homogeneity",
    "hue",
    "saturation",
    "value",
]


def foreground_pixel_mask(bgr_img, gray, hsv):
    """Mask out dark shadow, white paper, and cool slate/navy background pixels."""
    b_ch, _, r_ch = cv2.split(bgr_img)
    return (
        (gray > 20)
        & ~((gray > 130) & (hsv[:, :, 1] < 25))
        & ~((b_ch.astype(int) >= r_ch.astype(int) - 2) & (gray < 145))
    )


def preprocess_image(img, target_size=FEATURE_PATCH_SIZE):
    """Convert to grayscale, apply Gaussian blur, and compute resized stats/histograms."""
    if img is None:
        raise ValueError("Invalid input image (None)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    resized_blur = cv2.GaussianBlur(resized_gray, (3, 3), 0)

    hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten().astype(int).tolist()
    hist_blur = cv2.calcHist([blur], [0], None, [32], [0, 256]).flatten().astype(int).tolist()

    return {
        "original": img,
        "gray": gray,
        "blur": blur,
        "resized_img": resized_img,
        "resized_gray": resized_gray,
        "resized_blur": resized_blur,
        "stats": {
            "gray_mean": round(float(gray.mean()), 2),
            "gray_std": round(float(gray.std()), 2),
            "blur_mean": round(float(blur.mean()), 2),
            "blur_std": round(float(blur.std()), 2),
            "hist_gray": hist_gray,
            "hist_blur": hist_blur,
        },
    }


def extract_features(kernel_img, is_preprocessed=False):
    """Extract 4 GLCM texture features and 3 background-compensated HSV color features."""
    if is_preprocessed and kernel_img.shape[:2] == FEATURE_PATCH_SIZE:
        smoothed = kernel_img
    else:
        resized = cv2.resize(kernel_img, FEATURE_PATCH_SIZE, interpolation=cv2.INTER_CUBIC)
        smoothed = cv2.GaussianBlur(resized, (3, 3), 0)

    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = float(graycoprops(glcm, "contrast").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())

    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    fg_mask = foreground_pixel_mask(smoothed, gray, hsv)
    if np.sum(fg_mask) <= 150:
        fg_mask = gray > 15

    if np.sum(fg_mask) > 20:
        h_mean = float(hsv[:, :, 0][fg_mask].mean())
        s_mean = float(hsv[:, :, 1][fg_mask].mean())
        v_mean = float(hsv[:, :, 2][fg_mask].mean())
    else:
        h_mean = float(hsv[:, :, 0].mean())
        s_mean = float(hsv[:, :, 1].mean())
        v_mean = float(hsv[:, :, 2].mean())

    return {
        "contrast": round(contrast, 4),
        "correlation": round(correlation, 4),
        "energy": round(energy, 4),
        "homogeneity": round(homogeneity, 4),
        "hue": round(h_mean, 2),
        "saturation": round(s_mean, 2),
        "value": round(v_mean, 2),
    }
