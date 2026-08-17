import numpy as np
import cv2
from core.filter import GuidedFilter
from cv.image import to_32F, to_8U


def estimate_dark_channel(image, radius=7):
    """
    Estimate the dark channel prior of an image.
    
    Parameters
    ----------
    image: NDArray
        HWC array of shape (H, W, 3) in float32, range [0, 1]
    radius: int
        Radius of the local patch window
        
    Returns
    -------
    dark: NDArray
        2D array of shape (H, W) representing the dark channel prior
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 2*radius+1))
    dark = cv2.erode(min_channel, kernel)
    return dark


def estimate_atmospheric_light(image, dark, percentage=0.001):
    """
    Estimate the atmospheric light A.
    
    Parameters
    ----------
    image: NDArray
        HWC array of shape (H, W, 3) in float32, range [0, 1]
    dark: NDArray
        2D dark channel prior map
    percentage: float
        Top percentage of brightest pixels in dark channel to consider
        
    Returns
    -------
    A: NDArray
        1D array of shape (3,) representing RGB atmospheric light
    """
    h, w, c = image.shape
    num_pixels = h * w
    num_brightest = max(1, int(num_pixels * percentage))
    
    dark_flat = dark.flatten()
    image_flat = image.reshape(-1, c)
    
    # Get indices of top brightest pixels in dark channel
    indices = np.argsort(dark_flat)[-num_brightest:]
    
    # Among these, select the pixel in the original image with highest intensity
    brightest_pixels = image_flat[indices]
    intensities = np.sum(brightest_pixels, axis=1)
    best_idx = np.argmax(intensities)
    
    A = brightest_pixels[best_idx]
    return A


def estimate_transmission(image, A, radius=7, omega=0.95):
    """
    Estimate the rough transmission map.
    
    Parameters
    ----------
    image: NDArray
        HWC array of shape (H, W, 3) in float32, range [0, 1]
    A: NDArray
        1D array of shape (3,) representing RGB atmospheric light
    radius: int
        Radius of local patch window
    omega: float
        Haze preservation factor (keeps a small amount of haze for depth)
        
    Returns
    -------
    transmission: NDArray
        2D transmission map of shape (H, W)
    """
    normalized = image / np.maximum(A, 1e-6)
    dark_normalized = estimate_dark_channel(normalized, radius)
    transmission = 1.0 - omega * dark_normalized
    return transmission


def dehaze(image, radius=15, eps=0.001, omega=0.95, t0=0.1, guided_radius=40, guided_eps=1e-3):
    """
    Remove haze from a single image using Guided Image Filtering.
    
    Parameters
    ----------
    image: NDArray
        HWC array representing the hazy image in range [0, 255] or [0, 1]
    radius: int
        Local window radius for DCP estimation
    eps: float
        Guided filter epsilon parameter
    omega: float
        Haze preservation factor
    t0: float
        Lower bound for transmission map
    guided_radius: int
        Radius for the guided filter refinement
    guided_eps: float
        Regularization parameter for guided filter refinement
        
    Returns
    -------
    dehazed: NDArray
        Haze-free image in float32, range [0, 1]
    """
    I = to_32F(image)
    dark = estimate_dark_channel(I, radius=radius)
    A = estimate_atmospheric_light(I, dark)
    t_rough = estimate_transmission(I, A, radius=radius, omega=omega)
    
    # Get guide image (grayscale version of input)
    if len(I.shape) == 3:
        guide = cv2.cvtColor(to_8U(I), cv2.COLOR_RGB2GRAY)
        guide = to_32F(guide)
    else:
        guide = I
        
    # Refine the rough transmission map using Guided Filter
    gf = GuidedFilter(guide, radius=guided_radius, eps=guided_eps)
    t_refined = gf.filter(t_rough)
    t_refined = np.clip(t_refined, t0, 1.0)
    
    # Recover scene radiance
    A_expanded = np.expand_dims(np.expand_dims(A, axis=0), axis=0)  # (1, 1, 3)
    t_expanded = np.expand_dims(t_refined, axis=2)  # (H, W, 1)
    
    J = (I - A_expanded) / t_expanded + A_expanded
    return np.clip(J, 0.0, 1.0)


def detail_enhancement(image, radius=8, eps=0.04, factor=3.0, color_space_preserve=True):
    """
    Enhance detail and sharpen the input image using Guided Filter.
    
    Parameters
    ----------
    image: NDArray
        Input image in range [0, 255] or [0, 1]
    radius: int
        Radius of guided filter
    eps: float
        Epsilon of guided filter (regularization)
    factor: float
        Detail boosting factor (e.g., 2.0 or 3.0)
    color_space_preserve: bool
        If True, applies enhancement only on the Y (Luminance) channel in YCrCb
        color space to avoid color shifts. Otherwise, applies channel-wise.
        
    Returns
    -------
    enhanced: NDArray
        Detail-enhanced image in float32, range [0, 1]
    """
    I = to_32F(image)
    if len(I.shape) == 3 and color_space_preserve:
        img_8u = to_8U(I)
        ycrcb = cv2.cvtColor(img_8u, cv2.COLOR_RGB2YCrCb)
        ycrcb = ycrcb.astype(np.float32) / 255.0
        
        y = ycrcb[:, :, 0]
        
        gf = GuidedFilter(y, radius=radius, eps=eps)
        base = gf.filter(y)
        detail = y - base
        enhanced_y = base + factor * detail
        enhanced_y = np.clip(enhanced_y, 0.0, 1.0)
        
        ycrcb[:, :, 0] = enhanced_y
        enhanced_8u = to_8U(ycrcb)
        enhanced_rgb = cv2.cvtColor(enhanced_8u, cv2.COLOR_YCrCb2RGB)
        return to_32F(enhanced_rgb)
    else:
        gf = GuidedFilter(I, radius=radius, eps=eps)
        base = gf.filter(I)
        detail = I - base
        enhanced = base + factor * detail
        return np.clip(enhanced, 0.0, 1.0)


def joint_filtering(image, guide, radius=4, eps=0.01):
    """
    Smooth a noisy/target image using a separate guide image.
    
    Parameters
    ----------
    image: NDArray
        Target image to filter
    guide: NDArray
        Guide image containing edge structures
    radius: int
        Radius of guided filter
    eps: float
        Regularization parameter
        
    Returns
    -------
    filtered: NDArray
        Filtered image in float32, range [0, 1]
    """
    I = to_32F(guide)
    p = to_32F(image)
    gf = GuidedFilter(I, radius=radius, eps=eps)
    return gf.filter(p)
