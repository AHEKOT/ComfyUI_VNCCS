"""Tests for nodes/pose_generator.py — pure helper functions."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("torch")

from nodes.pose_generator import _clamp, _apply_safe_zone, _sanitize_joints, CANVAS_WIDTH, CANVAS_HEIGHT


# ── _clamp ────────────────────────────────────────────────────────────────────

class TestClamp:
    def test_below_minimum(self):
        assert _clamp(-5, 0, 100) == 0

    def test_above_maximum(self):
        assert _clamp(150, 0, 100) == 100

    def test_within_range(self):
        assert _clamp(50, 0, 100) == 50

    def test_at_minimum(self):
        assert _clamp(0, 0, 100) == 0

    def test_at_maximum(self):
        assert _clamp(100, 0, 100) == 100

    def test_negative_range(self):
        assert _clamp(-3, -10, -1) == -3

    def test_single_value_range(self):
        assert _clamp(5, 7, 7) == 7


# ── _apply_safe_zone ──────────────────────────────────────────────────────────

class TestApplySafeZone:
    def _simple_pose(self, x, y):
        return {"nose": (x, y), "neck": (x, y + 50)}

    def test_scale_1_unchanged(self):
        poses = [self._simple_pose(100, 200)]
        result = _apply_safe_zone(poses, 1.0, 256, 768)
        assert result[0]["nose"] == (100, 200)
        assert result[0]["neck"] == (100, 250)

    def test_scale_0_snaps_to_center(self):
        cx, cy = 256, 768
        poses = [{"nose": (100, 200)}]
        result = _apply_safe_zone(poses, 0.0, cx, cy)
        assert result[0]["nose"] == (cx, cy)

    def test_output_clamped_to_canvas(self):
        poses = [{"nose": (0, 0)}]
        result = _apply_safe_zone(poses, 2.0, 256, 768)
        x, y = result[0]["nose"]
        assert 0 <= x < CANVAS_WIDTH
        assert 0 <= y < CANVAS_HEIGHT

    def test_non_tuple_coords_passed_through(self):
        poses = [{"unknown_field": "string_value"}]
        result = _apply_safe_zone(poses, 0.5, 256, 768)
        assert result[0]["unknown_field"] == "string_value"

    def test_multiple_poses_all_scaled(self):
        poses = [{"nose": (100, 100)}, {"nose": (400, 500)}]
        result = _apply_safe_zone(poses, 0.5, 256, 768)
        assert len(result) == 2
        # All joints should be scaled toward center
        x0, y0 = result[0]["nose"]
        assert 0 <= x0 < CANVAS_WIDTH

    def test_returns_list_same_length(self):
        poses = [self._simple_pose(100, 100)] * 5
        result = _apply_safe_zone(poses, 0.8, 256, 768)
        assert len(result) == 5


# ── _sanitize_joints ──────────────────────────────────────────────────────────

class TestSanitizeJoints:
    def test_valid_joint_preserved(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        # Use first joint from DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        result = _sanitize_joints({joint: (100, 200)})
        assert joint in result
        assert result[joint] == (100, 200)

    def test_unknown_joint_ignored(self):
        result = _sanitize_joints({"__totally_invalid_joint__": (100, 200)})
        assert "__totally_invalid_joint__" not in result

    def test_missing_joints_filled_with_defaults(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        result = _sanitize_joints({})
        for joint in DEFAULT_SKELETON:
            assert joint in result

    def test_out_of_bounds_clamped(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        result = _sanitize_joints({joint: (99999, 99999)})
        x, y = result[joint]
        assert x <= CANVAS_WIDTH - 1
        assert y <= CANVAS_HEIGHT - 1

    def test_negative_coords_clamped(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        result = _sanitize_joints({joint: (-100, -200)})
        x, y = result[joint]
        assert x >= 0
        assert y >= 0

    def test_float_coords_rounded_to_int(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        result = _sanitize_joints({joint: (100.7, 200.3)})
        x, y = result[joint]
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert x == 101
        assert y == 200

    def test_list_coords_accepted(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        result = _sanitize_joints({joint: [150, 300]})
        assert result[joint] == (150, 300)

    def test_missing_coords_skipped(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        result = _sanitize_joints({joint: (100,)})  # only 1 coord
        # Should not crash, but may use default
        assert joint in result

    def test_non_numeric_coords_skipped(self):
        from nodes.pose_generator import DEFAULT_SKELETON
        joint = next(iter(DEFAULT_SKELETON))
        before = _sanitize_joints({})
        result = _sanitize_joints({joint: ("a", "b")})
        # Should fall back to default
        assert result[joint] == before[joint]
