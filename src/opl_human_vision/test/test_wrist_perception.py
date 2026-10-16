from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from wrist_perception import (  # noqa: E402
    classify_color,
    depth_at_box_center,
    depth_to_meters,
    timestamps_within_slop,
    valid_intrinsics,
)


def image(data, encoding, width, height, step, bigendian=False):
    return SimpleNamespace(
        data=data, encoding=encoding, width=width, height=height, step=step,
        is_bigendian=bigendian)


@pytest.mark.parametrize(('hue', 'expected'), [
    (0, 'red'), (30, 'yellow'), (60, 'green'), (110, 'blue')])
def test_dominant_hsv_color(hue, expected):
    hsv = np.full((20, 20, 3), (hue, 220, 220), dtype=np.uint8)
    assert classify_color(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)) == expected


def test_color_falls_back_to_unknown_for_gray_and_unmatched_hues():
    gray = np.full((10, 10, 3), 130, dtype=np.uint8)
    purple_hsv = np.full((10, 10, 3), (145, 220, 220), dtype=np.uint8)
    assert classify_color(gray) == 'unknown'
    assert classify_color(cv2.cvtColor(purple_hsv, cv2.COLOR_HSV2BGR)) == 'unknown'


def test_depth_32fc1_handles_row_padding_and_invalid_pixels():
    rows = np.array([[1.25, np.nan], [0.0, 2.5]], dtype='<f4')
    padded = rows.tobytes()[:8] + b'PAD!' + rows.tobytes()[8:] + b'PAD!'
    decoded = depth_to_meters(image(padded, '32FC1', 2, 2, 12))
    assert decoded[0, 0] == pytest.approx(1.25)
    assert np.isnan(decoded[0, 1])
    assert decoded[1, 0] == 0
    assert depth_at_box_center(decoded, 0.5, 0.5, 2, 2) is None


def test_depth_16uc1_scales_millimeters_to_meters_and_honors_endianness():
    values = np.array([[1250, 0], [2300, 1000]], dtype='>u2')
    decoded = depth_to_meters(image(values.tobytes(), '16UC1', 2, 2, 4, True))
    assert decoded[0, 0] == pytest.approx(1.25)
    assert depth_at_box_center(decoded, 0.5, 0.5, 2, 2) == pytest.approx(1.25)


@pytest.mark.parametrize('bad', [
    image(b'\x00' * 4, '8UC1', 2, 2, 2),
    image(b'\x00' * 4, '32FC1', 2, 2, 8),
    image(b'\x00' * 16, '32FC1', 2, 2, 4),
])
def test_depth_rejects_unsupported_or_malformed_messages(bad):
    with pytest.raises(ValueError):
        depth_to_meters(bad)


def test_sync_requires_nonzero_stamps_within_configured_slop():
    assert timestamps_within_slop([1.0, 1.02, 1.01, 1.0], 0.03)
    assert not timestamps_within_slop([1.0, 1.04], 0.03)
    assert not timestamps_within_slop([1.0, 0.0], 0.03)


def test_camera_intrinsics_must_be_finite_and_nonzero():
    info = SimpleNamespace(width=640, height=480, k=[500, 0, 320, 0, 500, 240, 0, 0, 1])
    assert valid_intrinsics(info) == (500.0, 500.0, 320.0, 240.0)
    info.k[0] = 0
    assert valid_intrinsics(info) is None


def test_detected_object_interface_carries_point_without_orientation():
    from opl_interfaces.msg import DetectedObject, DetectedObjectArray

    assert set(DetectedObject.get_fields_and_field_types()) == {
        'class_name', 'color', 'confidence', 'position'}
    assert set(DetectedObjectArray.get_fields_and_field_types()) == {'header', 'objects'}
    assert 'Pose' not in repr(DetectedObject.get_fields_and_field_types())
