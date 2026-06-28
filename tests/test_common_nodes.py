"""Tests for nodes/common_nodes.py."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nodes.common_nodes import (
    VNCCS_Integer, VNCCS_Float, VNCCS_String,
    VNCCS_MultilineText, VNCCS_PromptConcat,
)


class TestVNCCSInteger:
    def test_returns_int(self):
        assert VNCCS_Integer().pass_through(42) == (42,)

    def test_unwraps_list(self):
        assert VNCCS_Integer().pass_through([7]) == (7,)

    def test_zero(self):
        assert VNCCS_Integer().pass_through(0) == (0,)

    def test_negative(self):
        assert VNCCS_Integer().pass_through(-5) == (-5,)

    def test_casts_float_to_int(self):
        assert VNCCS_Integer().pass_through(3.9) == (3,)


class TestVNCCSFloat:
    def test_returns_float(self):
        assert VNCCS_Float().pass_through(3.14) == (3.14,)

    def test_unwraps_list(self):
        assert VNCCS_Float().pass_through([2.5]) == (2.5,)

    def test_int_becomes_float(self):
        result = VNCCS_Float().pass_through(1)
        assert result == (1.0,)
        assert isinstance(result[0], float)

    def test_zero(self):
        assert VNCCS_Float().pass_through(0.0) == (0.0,)


class TestVNCCSString:
    def test_return_type_tolerates_legacy_high_slot_validation(self):
        assert VNCCS_String.RETURN_TYPES[3] == "STRING"

    def test_returns_string(self):
        assert VNCCS_String().pass_through("hello") == ("hello",)

    def test_unwraps_list(self):
        assert VNCCS_String().pass_through(["world"]) == ("world",)

    def test_empty_string(self):
        assert VNCCS_String().pass_through("") == ("",)

    def test_casts_non_string(self):
        assert VNCCS_String().pass_through(42) == ("42",)


class TestVNCCSMultilineText:
    def test_preserves_newlines(self):
        assert VNCCS_MultilineText().pass_through("line1\nline2") == ("line1\nline2",)

    def test_unwraps_list(self):
        assert VNCCS_MultilineText().pass_through(["a\nb"]) == ("a\nb",)

    def test_empty(self):
        assert VNCCS_MultilineText().pass_through("") == ("",)


class TestVNCCSPromptConcat:
    def test_all_parts(self):
        result = VNCCS_PromptConcat().concat("a", "b", "c", "d", ",")
        assert result == ("a,b,c,d",)

    def test_skips_empty(self):
        result = VNCCS_PromptConcat().concat("a", "", "c", "", ",")
        assert result == ("a,c",)

    def test_custom_separator(self):
        result = VNCCS_PromptConcat().concat("A", "B", "C", "D", " | ")
        assert result == ("A | B | C | D",)

    def test_all_empty(self):
        result = VNCCS_PromptConcat().concat("", "", "", "", ",")
        assert result == ("",)

    def test_unwraps_list_inputs(self):
        result = VNCCS_PromptConcat().concat(["a"], ["b"], ["c"], ["d"], [","])
        assert result == ("a,b,c,d",)

    def test_strips_whitespace_from_parts(self):
        result = VNCCS_PromptConcat().concat("  a  ", "  b  ", "", "", ",")
        assert result == ("a,b",)

    def test_single_part(self):
        result = VNCCS_PromptConcat().concat("only", "", "", "", ",")
        assert result == ("only",)
