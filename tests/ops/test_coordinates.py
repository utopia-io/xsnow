# tests/features/test_coordinates.py
"""
Tests for coordinate handling: sanitization, duplicates, string coordinates.
Focus: Core coordinate functionality independent of I/O.
"""
import pytest
import numpy as np
from xsnow.io.base import _handle_duplicate_names, _sanitize_station_name


class TestStationNameSanitization:
    """Test station name sanitization."""
    
    def test_sanitize_normal_name(self):
        assert _sanitize_station_name("VIR1A") == "VIR1A"
        assert _sanitize_station_name("Station_001") == "Station_001"
    
    def test_sanitize_whitespace(self):
        assert _sanitize_station_name("  VIR1A  ") == "VIR1A"
        assert _sanitize_station_name("Station A") == "Station_A"
    
    def test_sanitize_special_characters(self):
        assert _sanitize_station_name("Station@#1") == "Station__1"
        assert _sanitize_station_name("Site/Test") == "Site_Test"
    
    def test_sanitize_empty_string(self):
        assert _sanitize_station_name("") == "UNNAMED_STATION"
        assert _sanitize_station_name("   ") == "UNNAMED_STATION"
    
    def test_sanitize_long_names(self):
        long_name = "A" * 100
        result = _sanitize_station_name(long_name)
        assert len(result) == 64


class TestDuplicateNameHandling:
    """Test duplicate station name handling."""
    
    def test_no_duplicates(self):
        names = ['A', 'B', 'C']
        assert _handle_duplicate_names(names) == names
    
    def test_simple_duplicates(self):
        names = ['A', 'B', 'A', 'C']
        result = _handle_duplicate_names(names)
        assert result == ['A_0', 'B', 'A_1', 'C']
    
    def test_multiple_duplicates(self):
        names = ['A', 'A', 'A', 'B']
        result = _handle_duplicate_names(names)
        assert result == ['A_0', 'A_1', 'A_2', 'B']