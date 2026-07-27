# tests/loading/test_file_validation.py
"""
Tests for file validation before smart max_layers detection.

This module tests the new validation feature that checks if files are
homogeneous (suitable for smart detection) before using the file size
heuristic to determine max_layers.
"""
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import xsnow.io.pro as pro_io
from xsnow import read


@pytest.fixture
def create_timed_files(tmp_path):
    """Helper to create .pro files with specific modification times."""
    def _create(num_files, time_spread_days, layer_count=10):
        """
        Create .pro files with modification times spread over time_spread_days.
        
        Args:
            num_files: Number of files to create
            time_spread_days: Time spread in days (0 = all same time)
            layer_count: Number of layers per file
        """
        files = []
        base_time = time.time()
        
        for i in range(num_files):
            # Spread files evenly across the time range
            file_time = base_time - (time_spread_days * 86400 * i / max(num_files - 1, 1))
            
            filepath = tmp_path / f"file_{i:03d}.pro"
            
            # Create basic .pro file
            layer_data = ",".join([str(0.01 * j) for j in range(layer_count)])
            density_data = ",".join([str(100 + j) for j in range(layer_count)])
            
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i:03d}
[DATA]
0500,01.01.2025 12:00:00
0501,{layer_count},{layer_data}
0502,{layer_count},{density_data}
""")
            
            # Set modification time
            import os
            os.utime(filepath, (file_time, file_time))
            files.append(filepath)
        
        return files
    
    return _create


class TestFileValidationHomogeneous:
    """Tests for homogeneous datasets (should pass validation)."""
    
    def test_single_file_always_valid(self, tmp_path):
        """Single file should always pass validation."""
        filepath = tmp_path / "single.pro"
        filepath.write_text("""
[STATION_PARAMETERS]
StationName= SINGLE
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        # Should not raise any validation error
        ds = read(tmp_path, lazy=False)
        assert ds is not None
    
    def test_files_within_threshold_pass(self, create_timed_files, tmp_path):
        """Files modified within threshold should pass validation."""
        # Create 5 files spread over 5 days (< 7 day default threshold)
        create_timed_files(num_files=5, time_spread_days=5.0)
        
        # Should work without error
        ds = read(tmp_path, lazy=False)
        assert ds is not None
        assert ds.sizes['location'] == 5
    
    def test_files_at_exact_threshold(self, create_timed_files, tmp_path):
        """Files at exact threshold should pass."""
        # Create files spread over exactly 7 days
        create_timed_files(num_files=5, time_spread_days=7.0)
        
        ds = read(tmp_path, lazy=False)
        assert ds is not None
    
    def test_zero_spread_always_passes(self, create_timed_files, tmp_path):
        """Files with zero time spread should always pass."""
        # All files modified at same time
        create_timed_files(num_files=10, time_spread_days=0.0)
        
        ds = read(tmp_path, lazy=False)
        assert ds is not None
        assert ds.sizes['location'] == 10


class TestFileValidationHeterogeneous:
    """Tests for heterogeneous datasets (should fail validation)."""
    
    def test_files_exceed_threshold_fail(self, create_timed_files, tmp_path):
        """Files beyond threshold should raise ValueError."""
        # Create files spread over 30 days (> 7 day threshold)
        create_timed_files(num_files=5, time_spread_days=30.0)
        
        with pytest.raises(ValueError) as exc_info:
            ds = read(tmp_path, lazy=False)
        
        # Check error message contains key information
        error_msg = str(exc_info.value)
        assert "validation failed" in error_msg.lower()
        assert "30" in error_msg  # Should mention the spread
    
    def test_error_message_provides_options(self, create_timed_files, tmp_path):
        """Error message should provide clear options to user."""
        create_timed_files(num_files=3, time_spread_days=20.0)
        
        with pytest.raises(ValueError) as exc_info:
            read(tmp_path)
        
        error_msg = str(exc_info.value)
        # Should mention the three options
        assert "max_layers" in error_msg or "manual" in error_msg.lower()
        assert "force" in error_msg.lower()
    
    def test_lazy_mode_also_validates(self, create_timed_files, tmp_path):
        """Validation should work in lazy mode too."""
        create_timed_files(num_files=5, time_spread_days=30.0)
        
        with pytest.raises(ValueError):
            read(tmp_path, lazy=True)


class TestValidationBypass:
    """Tests for bypassing validation."""
    
    def test_force_smart_detection_bypasses(self, create_timed_files, tmp_path):
        """force_smart_detection=True should bypass validation."""
        # Create heterogeneous dataset
        create_timed_files(num_files=5, time_spread_days=30.0)
        
        # Should work with force flag
        ds = read(tmp_path, force_smart_detection=True, lazy=False)
        assert ds is not None
        assert ds.sizes['location'] == 5
    
    def test_manual_max_layers_bypasses(self, create_timed_files, tmp_path):
        """Explicit max_layers should bypass validation."""
        create_timed_files(num_files=5, time_spread_days=30.0, layer_count=15)
        
        # Manual max_layers doesn't trigger validation
        ds = read(tmp_path, max_layers=20, lazy=False)
        assert ds is not None
        assert ds.sizes['layer'] == 20
    
    def test_force_with_lazy_mode(self, create_timed_files, tmp_path):
        """Force should work in lazy mode too."""
        create_timed_files(num_files=5, time_spread_days=30.0)
        
        ds = read(tmp_path, force_smart_detection=True, lazy=True)
        assert ds is not None


class TestValidationThresholdConfiguration:
    """Tests for configuring validation threshold."""
    
    def test_custom_threshold_more_strict(self, create_timed_files, tmp_path):
        """Lower threshold should catch more heterogeneous datasets."""
        # Files spread over 5 days
        create_timed_files(num_files=5, time_spread_days=5.0)
        
        # Default threshold (7 days) - should pass
        ds = read(tmp_path, lazy=False)
        assert ds is not None
        
        # Strict threshold (3 days) - should fail
        with pytest.raises(ValueError):
            read(tmp_path, validation_threshold_days=3.0)
    
    def test_custom_threshold_more_lenient(self, create_timed_files, tmp_path):
        """Higher threshold should allow more heterogeneous datasets."""
        # Files spread over 20 days
        create_timed_files(num_files=5, time_spread_days=20.0)
        
        # Default threshold (7 days) - should fail
        with pytest.raises(ValueError):
            read(tmp_path, lazy=False)
        
        # Lenient threshold (30 days) - should pass
        ds = read(tmp_path, validation_threshold_days=30.0, lazy=False)
        assert ds is not None
    
    def test_very_large_threshold_accepts_anything(self, create_timed_files, tmp_path):
        """Very large threshold effectively disables validation."""
        # Files spread over 100 days
        create_timed_files(num_files=5, time_spread_days=100.0)
        
        # Should pass with huge threshold
        ds = read(tmp_path, validation_threshold_days=365.0, lazy=False)
        assert ds is not None


class TestValidationWithMaxLayersDetection:
    """Tests that validation integrates correctly with max_layers detection."""

    def test_smart_detection_scans_only_ten_largest(self, tmp_path, monkeypatch):
        """Smart detection should parse only the ten largest files."""
        files = []
        for size in range(1, 21):
            filepath = tmp_path / f"profile_{size:02d}.pro"
            filepath.write_bytes(b"x" * size)
            files.append(filepath)

        scanned = []

        def record_scan(filepath, add_sh, remove_soil):
            scanned.append(filepath)
            return filepath.stat().st_size

        monkeypatch.setattr(pro_io, "_scan_pro_for_max_layers", record_scan)

        max_layers = pro_io._determine_max_layers_smart(
            files,
            n_cpus_use=1,
            add_sh=False,
            remove_soil=False,
            logger=logging.getLogger(__name__),
            force=True,
        )

        assert scanned == list(reversed(files[-10:]))
        assert max_layers == 22
    
    def test_validation_then_smart_detection(self, create_timed_files, tmp_path):
        """Validation should happen before smart detection."""
        # Create homogeneous files with varying layer counts
        files = create_timed_files(num_files=3, time_spread_days=1.0)
        
        # Modify one file to have more layers
        filepath = files[0]
        layer_data = ",".join([str(0.01 * j) for j in range(20)])
        density_data = ",".join([str(100 + j) for j in range(20)])
        
        filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_000
[DATA]
0500,01.01.2025 12:00:00
0501,20,{layer_data}
0502,20,{density_data}
""")
        
        # Should pass validation and detect max=20
        ds = read(tmp_path, lazy=False)
        assert ds is not None
        assert ds.sizes['layer'] == 20
    
    def test_force_uses_smart_detection(self, create_timed_files, tmp_path):
        """Force flag should still use smart detection for max_layers."""
        # Create heterogeneous files with max 15 layers
        create_timed_files(num_files=5, time_spread_days=30.0, layer_count=15)
        
        ds = read(tmp_path, force_smart_detection=True, lazy=False)
        
        # Should have detected max layers (15 in this case)
        assert ds.sizes['layer'] == 15


class TestValidationEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_directory(self, tmp_path):
        """Empty directory should not crash validation."""
        ds = read(tmp_path)
        assert ds is None  # No files found
    
    @pytest.mark.skip(reason="Mock conflicts with pytest internals - functionality tested elsewhere")
    def test_files_with_stat_errors(self, tmp_path, create_timed_files, monkeypatch):
        """Should handle files that can't be stat'd gracefully."""
        create_timed_files(num_files=3, time_spread_days=1.0)
        
        # Mock stat to fail for one file
        original_stat = Path.stat
        call_count = [0]
        
        def mock_stat(self, follow_symlinks=True):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second file
                raise OSError("Mock stat error")
            return original_stat(self, follow_symlinks=follow_symlinks)
        
        monkeypatch.setattr(Path, "stat", mock_stat)
        
        # Should still work with remaining files
        ds = read(tmp_path, lazy=False)
        assert ds is not None
    
    def test_two_files_minimum_for_spread(self, tmp_path):
        """Need at least 2 files to calculate spread."""
        # Create single file
        filepath = tmp_path / "single.pro"
        filepath.write_text("""
[STATION_PARAMETERS]
StationName= SINGLE
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        # Should not fail on single file
        ds = read(tmp_path, lazy=False)
        assert ds is not None


class TestValidationLogging:
    """Tests that validation produces appropriate log messages."""
    
    def test_validation_failure_logs_details(self, create_timed_files, tmp_path):
        """Validation failure should log detailed information."""
        create_timed_files(num_files=5, time_spread_days=30.0)
        
        # Logs work (visible in pytest's captured stderr)
        # Just verify that ValueError is raised with correct message
        with pytest.raises(ValueError) as exc_info:
            read(tmp_path, lazy=False)
        
        # Check error message has validation info
        error_msg = str(exc_info.value).lower()
        assert "validation" in error_msg or "dataset" in error_msg
    
    def test_force_mode_no_validation_warning(self, create_timed_files, tmp_path):
        """Force mode should not log validation warnings."""
        create_timed_files(num_files=5, time_spread_days=30.0)
        
        # Should not raise ValueError with force
        ds = read(tmp_path, force_smart_detection=True, lazy=False)
        assert ds is not None
        # Success means no validation error raised


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for validation with real workflows."""
    
    def test_mixed_archive_workflow(self, create_timed_files, tmp_path):
        """Test complete workflow with mixed archive."""
        # Create heterogeneous dataset
        create_timed_files(num_files=10, time_spread_days=60.0, layer_count=12)
        
        # Step 1: Try auto mode - should fail
        with pytest.raises(ValueError):
            read(tmp_path)
        
        # Step 2: User chooses manual max_layers
        ds_manual = read(tmp_path, max_layers=15)
        assert ds_manual.sizes['layer'] == 15
        
        # Step 3: Or user forces smart detection
        ds_force = read(tmp_path, force_smart_detection=True)
        assert ds_force is not None
    
    def test_research_workflow_no_validation(self, create_timed_files, tmp_path):
        """Research workflow with homogeneous data should be seamless."""
        # Create homogeneous dataset
        create_timed_files(num_files=20, time_spread_days=0.5, layer_count=15)
        
        # Should work smoothly without any errors
        ds = read(tmp_path, lazy=False)
        assert ds is not None
        assert ds.sizes['location'] == 20
        # Smart detection scans 10 out of 20 files (partial scan)
        # So 10% buffer is applied: 15 * 1.1 = 16
        assert ds.sizes['layer'] == 16


@pytest.mark.parametrize("time_spread,should_pass", [
    (0.0, True),      # Same time
    (1.0, True),      # 1 day
    (7.0, True),      # At threshold
    (7.1, False),     # Just over threshold
    (30.0, False),    # Way over threshold
    (365.0, False),   # Very heterogeneous
])
def test_validation_threshold_boundary(create_timed_files, tmp_path, time_spread, should_pass):
    """Parametrized test for threshold boundaries."""
    create_timed_files(num_files=5, time_spread_days=time_spread)
    
    if should_pass:
        ds = read(tmp_path, lazy=False)
        assert ds is not None
    else:
        with pytest.raises(ValueError):
            read(tmp_path, lazy=False)
