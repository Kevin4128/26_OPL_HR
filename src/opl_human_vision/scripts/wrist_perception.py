"""ROS-independent image and timestamp helpers for wrist object perception."""

import numpy as np


def timestamps_within_slop(stamps, slop):
    """Return true when all finite acquisition timestamps are mutually close."""
    values = [float(value) for value in stamps]
    return (bool(values) and all(np.isfinite(values)) and all(value > 0 for value in values)
            and max(values) - min(values) <= slop)


def depth_to_meters(image):
    """Decode a ROS Image payload with row padding and declared endianness."""
    if image.encoding == '32FC1':
        dtype = np.dtype('>f4' if image.is_bigendian else '<f4')
        scale = 1.0
    elif image.encoding == '16UC1':
        dtype = np.dtype('>u2' if image.is_bigendian else '<u2')
        scale = 0.001
    else:
        raise ValueError(f'Unsupported depth encoding: {image.encoding}')
    if image.height <= 0 or image.width <= 0 or image.step < image.width * dtype.itemsize:
        raise ValueError('Invalid depth dimensions or row step')
    if len(image.data) < image.height * image.step:
        raise ValueError('Depth payload is shorter than its dimensions and row step')
    values = np.ndarray(
        (image.height, image.width), dtype=dtype, buffer=image.data,
        strides=(image.step, dtype.itemsize))
    return values.astype(np.float32) * scale


def valid_intrinsics(camera_info):
    if camera_info.width <= 0 or camera_info.height <= 0 or len(camera_info.k) != 9:
        return None
    fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        return None
    return float(fx), float(fy), float(cx), float(cy)


def classify_color(bgr_crop):
    """Classify the dominant saturated color using the MCP crop thresholds."""
    import cv2

    if bgr_crop is None or bgr_crop.size == 0:
        return 'unknown'
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hues = hsv[:, :, 0]
    saturated = (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 45)
    if int(np.count_nonzero(saturated)) < max(4, int(saturated.size * 0.08)):
        return 'unknown'
    hue = hues[saturated]
    ranges = {
        'red': (hue <= 10) | (hue >= 170),
        'yellow': (hue > 23) & (hue <= 35),
        'green': (hue > 35) & (hue <= 85),
        'blue': (hue > 85) & (hue <= 130),
    }
    color, mask = max(ranges.items(), key=lambda item: int(np.count_nonzero(item[1])))
    return color if np.count_nonzero(mask) / hue.size >= 0.28 else 'unknown'


def depth_at_box_center(depth_m, center_x, center_y, width, height, min_valid=3):
    """Return the median finite depth in the central 30% of the detection box."""
    image_h, image_w = depth_m.shape
    half_w = max(1, int(width * 0.15))
    half_h = max(1, int(height * 0.15))
    cx, cy = int(round(center_x)), int(round(center_y))
    x1, x2 = max(0, cx - half_w), min(image_w, cx + half_w + 1)
    y1, y2 = max(0, cy - half_h), min(image_h, cy + half_h + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    values = depth_m[y1:y2, x1:x2]
    valid = values[np.isfinite(values) & (values > 0)]
    return float(np.median(valid)) if valid.size >= min_valid else None
