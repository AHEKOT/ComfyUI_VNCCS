"""Tests for nodes/sheet_manager.py — sheet split/compose and quad splitter."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.sheet_manager import VNCCSSheetManager, VNCCSSheetExtractor, VNCCS_QuadSplitter


def _sheet(rows=2, cols=6, cell_h=32, cell_w=16, ch=3):
    """Create a (rows*cell_h, cols*cell_w, ch) float tensor."""
    return torch.rand(rows * cell_h, cols * cell_w, ch)


def _batch(n, h, w, ch=3):
    return torch.rand(n, h, w, ch)


# ── VNCCSSheetManager.split_sheet ─────────────────────────────────────────────

class TestSplitSheet:
    def test_returns_12_parts(self):
        sheet = _sheet()
        parts = VNCCSSheetManager().split_sheet(sheet, 16, 32)
        assert len(parts) == 12

    def test_parts_resized_to_target(self):
        sheet = _sheet(cell_h=64, cell_w=32)
        parts = VNCCSSheetManager().split_sheet(sheet, 16, 32)
        for p in parts:
            assert p.shape[0] == 32
            assert p.shape[1] == 16

    def test_no_resize_when_sizes_match(self):
        sheet = _sheet(cell_h=32, cell_w=16)
        parts = VNCCSSheetManager().split_sheet(sheet, 16, 32)
        for p in parts:
            assert p.shape == (32, 16, 3)

    def test_channel_count_preserved(self):
        sheet = _sheet(ch=4)
        parts = VNCCSSheetManager().split_sheet(sheet, 16, 32)
        for p in parts:
            assert p.shape[2] == 4


# ── VNCCSSheetManager.compose_sheet ──────────────────────────────────────────

class TestComposeSheet:
    def test_returns_batched_tensor(self):
        imgs = _batch(12, 32, 16)
        result = VNCCSSheetManager().compose_sheet(imgs, 64)
        assert len(result.shape) == 4
        assert result.shape[0] == 1

    def test_pads_when_fewer_than_12(self):
        imgs = _batch(4, 32, 16)
        result = VNCCSSheetManager().compose_sheet(imgs, 64)
        assert result.shape[0] == 1

    def test_truncates_when_more_than_12(self):
        imgs = _batch(20, 32, 16)
        result = VNCCSSheetManager().compose_sheet(imgs, 64)
        assert result.shape[0] == 1

    def test_safe_margin_green_background(self):
        imgs = _batch(12, 32, 16)
        result = VNCCSSheetManager().compose_sheet(imgs, 64, safe_margin=True)
        # Green channel should be 1.0 somewhere in border area
        sheet = result[0]
        assert sheet[:, :, 1].max() >= 1.0


# ── VNCCSSheetManager.process_sheet ──────────────────────────────────────────

class TestProcessSheet:
    def test_split_mode_returns_list(self):
        sheet = torch.rand(1, 64, 96, 3)
        result, = VNCCSSheetManager().process_sheet("split", sheet, 16, 32)
        assert isinstance(result, list)
        assert len(result) == 12

    def test_compose_mode_returns_list_of_one(self):
        imgs = torch.rand(12, 32, 16, 3)
        result, = VNCCSSheetManager().process_sheet("compose", imgs, 16, 64)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            VNCCSSheetManager().process_sheet("invalid", torch.rand(1, 64, 96, 3), 16, 32)

    def test_list_inputs_unwrapped(self):
        sheet = torch.rand(1, 64, 96, 3)
        result, = VNCCSSheetManager().process_sheet(["split"], [sheet], [16], [32])
        assert len(result) == 12


# ── VNCCSSheetExtractor ───────────────────────────────────────────────────────

class TestSheetExtractor:
    def test_extracts_first_part(self):
        sheet = torch.rand(1, 64, 96, 3)
        result, = VNCCSSheetExtractor().extract(sheet, 1, 16, 32)
        assert result.shape[0] == 1

    def test_extracts_last_part(self):
        sheet = torch.rand(1, 64, 96, 3)
        result, = VNCCSSheetExtractor().extract(sheet, 12, 16, 32)
        assert result.shape[0] == 1

    def test_index_clamped_above(self):
        sheet = torch.rand(1, 64, 96, 3)
        result, = VNCCSSheetExtractor().extract(sheet, 99, 16, 32)
        assert result.shape[0] == 1

    def test_output_resized(self):
        sheet = torch.rand(1, 128, 192, 3)
        result, = VNCCSSheetExtractor().extract(sheet, 1, 8, 16)
        assert result.shape[2] == 8
        assert result.shape[1] == 16

    def test_different_parts_differ(self):
        # Fill each cell with a unique value
        sheet = torch.zeros(64, 96, 3)
        cell_h, cell_w = 32, 16
        for row in range(2):
            for col in range(6):
                sheet[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w, :] = (row * 6 + col) / 12.0
        sheet = sheet.unsqueeze(0)
        p1, = VNCCSSheetExtractor().extract(sheet, 1, 16, 32)
        p2, = VNCCSSheetExtractor().extract(sheet, 2, 16, 32)
        assert not torch.allclose(p1, p2)


# ── VNCCS_QuadSplitter ────────────────────────────────────────────────────────

class TestQuadSplitter:
    def test_split_returns_4_parts(self):
        img = torch.rand(1, 1, 100, 100, 3)  # wrong shape; wrap correctly
        img = torch.rand(1, 100, 100, 3)
        result, = VNCCS_QuadSplitter().split(img)
        assert len(result) == 4

    def test_split_square_each_part_is_half(self):
        img = torch.rand(1, 100, 100, 3)
        result, = VNCCS_QuadSplitter().split(img)
        for q in result:
            assert q.shape[1] == 50
            assert q.shape[2] == 50

    def test_split_non_square_center_cropped(self):
        img = torch.rand(1, 80, 100, 3)  # wider than tall
        result, = VNCCS_QuadSplitter().split(img)
        assert len(result) == 4
        # Each part should be square
        for q in result:
            assert q.shape[1] == q.shape[2]

    def test_center_crop_square_preserves_square(self):
        img = torch.rand(64, 64, 3)
        result = VNCCS_QuadSplitter()._center_crop_square(img)
        assert result.shape == (64, 64, 3)

    def test_center_crop_square_wide_image(self):
        img = torch.rand(50, 80, 3)
        result = VNCCS_QuadSplitter()._center_crop_square(img)
        assert result.shape[0] == result.shape[1] == 50

    def test_center_crop_square_tall_image(self):
        img = torch.rand(80, 50, 3)
        result = VNCCS_QuadSplitter()._center_crop_square(img)
        assert result.shape[0] == result.shape[1] == 50

    def test_normalize_image_list_from_batch(self):
        batch = torch.rand(3, 32, 32, 3)
        result = VNCCS_QuadSplitter()._normalize_image_list(batch)
        assert len(result) == 3
        for t in result:
            assert t.shape == (32, 32, 3)

    def test_normalize_image_list_nested(self):
        t1 = torch.rand(1, 32, 32, 3)
        t2 = torch.rand(1, 32, 32, 3)
        result = VNCCS_QuadSplitter()._normalize_image_list([[t1], [t2]])
        assert len(result) == 2

    def test_compose_4_into_grid(self):
        quads = [torch.rand(1, 50, 50, 3) for _ in range(4)]
        result, = VNCCS_QuadSplitter().compose(quads)
        assert len(result) == 1
        composite = result[0]
        assert composite.shape[1] == 100
        assert composite.shape[2] == 100

    def test_compose_fewer_than_4_raises(self):
        quads = [torch.rand(1, 50, 50, 3) for _ in range(3)]
        with pytest.raises(ValueError):
            VNCCS_QuadSplitter().compose(quads)

    def test_process_split_mode(self):
        img = torch.rand(1, 100, 100, 3)
        result, = VNCCS_QuadSplitter().process(["split"], [img])
        assert len(result) == 4

    def test_process_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            VNCCS_QuadSplitter().process(["bad"], [torch.rand(1, 100, 100, 3)])
