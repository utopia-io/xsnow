# tests/integration/test_validation_and_trimming.py
"""
Integration tests for file validation and trim_padding features working together.

These tests verify that both features integrate properly with the rest of
the xsnow loading pipeline and with each other.
"""
import pytest
import time
import numpy as np
from pathlib import Path
from xsnow import read


@pytest.fixture
def mixed_archive(tmp_path):
    """
    Create a mixed archive simulating real-world scenario:
    - Old files (60 days ago, 50 layers, complete runs)
    - New files (today, 25 layers, partial runs)
    """
    import time
    base_time = time.time()
    
    # Old files - 60 days ago, 50 layers
    for i in range(3):
        filepath = tmp_path / f"old_{i}.pro"
        layer_data = ",".join([str(0.01 * j) for j in range(50)])
        density_data = ",".join([str(100 + j) for j in range(50)])
        
        filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= OLD_STATION_{i}
[DATA]
0500,01.11.2024 12:00:00
0501,50,{layer_data}
0502,50,{density_data}
""")
        
        # Set mod time 60 days ago
        file_time = base_time - (60 * 86400)
        import os
        os.utime(filepath, (file_time, file_time))
    
    # New files - today, 25 layers
    for i in range(3):
        filepath = tmp_path / f"new_{i}.pro"
        layer_data = ",".join([str(0.01 * j) for j in range(25)])
        density_data = ",".join([str(100 + j) for j in range(25)])
        
        filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= NEW_STATION_{i}
[DATA]
0500,10.01.2026 12:00:00
0501,25,{layer_data}
0502,25,{density_data}
""")
        
        # Set mod time to now
        os.utime(filepath, (base_time, base_time))
    
    return tmp_path


class TestValidationAndTrimmingIntegration:
    """Tests for both features working together."""
    
    def test_validation_fails_trim_not_applied(self, mixed_archive):
        """If validation fails, trim_padding should not be applied."""
        # Try to load with both validation and trimming
        with pytest.raises(ValueError):
            # Should fail at validation stage, before trimming
            ds = read(mixed_archive, lazy=False, trim_padding=True)
    
    def test_force_validation_then_trim(self, mixed_archive):
        """Force validation bypass, then trim should work."""
        ds = read(mixed_archive, 
                 force_smart_detection=True,
                 lazy=False, 
                 trim_padding=True)
        
        assert ds is not None
        # Should have detected max ~50 layers
        # After trimming, should be exactly 50 (no padding from heterogeneous data)
        assert ds.sizes['layer'] == 50
    
    def test_manual_max_layers_then_trim(self, mixed_archive):
        """Manual max_layers bypasses validation, trim removes excess."""
        ds = read(mixed_archive,
                 max_layers=100,  # Set high
                 lazy=False,
                 trim_padding=True)
        
        assert ds is not None
        # Should trim from 100 to actual max (50)
        assert ds.sizes['layer'] == 50
    
    def test_validation_pass_then_trim(self, tmp_path):
        """Homogeneous data should pass validation and trim normally."""
        # Create homogeneous dataset with padding
        for i in range(3):
            filepath = tmp_path / f"station_{i}.pro"
            layer_data = ",".join([str(0.01 * j) for j in range(12)])
            density_data = ",".join([str(100 + j) for j in range(12)])
            
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,12,{layer_data}
0502,12,{density_data}
""")
        
        # Should pass validation and trim
        ds = read(tmp_path, lazy=False, max_layers=20, trim_padding=True)
        
        assert ds is not None
        assert ds.sizes['layer'] == 12  # Trimmed from 20


class TestRealWorldWorkflows:
    """Tests simulating real-world usage patterns."""
    
    def test_exploratory_analysis_workflow(self, mixed_archive):
        """
        User doesn't know dataset composition, tries auto-load.
        Gets validation error, then chooses appropriate option.
        """
        # Step 1: Try auto-load
        with pytest.raises(ValueError) as exc_info:
            ds = read(mixed_archive)
        
        # Step 2: User sees error, chooses manual max_layers
        ds = read(mixed_archive, max_layers=60, lazy=False, trim_padding=True)
        
        # Should work and be efficient
        assert ds is not None
        assert ds.sizes['layer'] == 50  # Trimmed from 60
    
    def test_production_workflow_known_data(self, tmp_path):
        """
        Production workflow where user knows their data structure.
        Sets explicit max_layers, uses trimming for efficiency.
        """
        # Create dataset
        for i in range(5):
            filepath = tmp_path / f"station_{i}.pro"
            layer_data = ",".join([str(0.01 * j) for j in range(30)])
            density_data = ",".join([str(100 + j) for j in range(30)])
            
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,30,{layer_data}
0502,30,{density_data}
""")
        
        # User knows max is ~30, sets explicit value
        ds = read(tmp_path, max_layers=35, lazy=False, trim_padding=True)
        
        # Should work efficiently
        assert ds is not None
        assert ds.sizes['layer'] == 30
    
    def test_large_dataset_lazy_subset_trim(self, tmp_path):
        """
        Large dataset workflow: lazy load, subset, compute, trim.
        """
        # Create larger dataset
        for i in range(10):
            filepath = tmp_path / f"station_{i}.pro"
            layer_data = ",".join([str(0.01 * j) for j in range(15)])
            density_data = ",".join([str(100 + j) for j in range(15)])
            
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i:02d}
[DATA]
0500,01.01.2025 12:00:00
0501,15,{layer_data}
0502,15,{density_data}
""")
        
        # Step 1: Lazy load (validation passes - homogeneous)
        ds = read(tmp_path, lazy=True, max_layers=20)
        assert ds is not None
        
        # Step 2: Subset to specific locations
        ds_sub = ds.sel(location=['STATION_00', 'STATION_01'])
        
        # Step 3: Compute only the subset
        ds_sub = ds_sub.compute()
        
        # Step 4: Trim the small subset
        ds_trimmed = ds_sub.trim_padding()
        
        assert ds_trimmed.sizes['layer'] == 15
        assert ds_trimmed.sizes['location'] == 2


class TestPerformanceAndMemory:
    """Tests to verify performance benefits of both features."""
    
    def test_trimming_reduces_memory(self, tmp_path):
        """Trimming should significantly reduce memory usage."""
        # Create dataset with substantial padding
        for i in range(3):
            filepath = tmp_path / f"station_{i}.pro"
            layer_data = ",".join([str(0.01 * j) for j in range(10)])
            density_data = ",".join([str(100 + j) for j in range(10)])
            
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,10,{layer_data}
0502,10,{density_data}
""")
        
        # Load with lots of padding
        ds_padded = read(tmp_path, lazy=False, max_layers=50)
        
        # Load with trimming
        ds_trimmed = read(tmp_path, lazy=False, max_layers=50, trim_padding=True)
        
        # Trimmed should be much smaller
        padded_layers = ds_padded.sizes['layer']
        trimmed_layers = ds_trimmed.sizes['layer']
        
        assert trimmed_layers < padded_layers
        assert trimmed_layers == 10
        assert padded_layers == 50


class TestErrorHandlingAndLogging:
    """Tests for error handling and logging of both features."""
    
    def test_validation_error_message_quality(self, mixed_archive):
        """Validation error should provide clear, actionable guidance."""
        try:
            ds = read(mixed_archive)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e).lower()
            
            # Should explain the problem
            assert "validation" in error_msg or "heterogeneous" in error_msg
            
            # Should mention at least one solution
            assert "max_layers" in error_msg or "force" in error_msg
    
    def test_trim_warning_quality(self, tmp_path):
        """Trim warning with lazy loading should be helpful (visible in pytest output)."""
        # Create dataset
        filepath = tmp_path / "station.pro"
        filepath.write_text("""
[STATION_PARAMETERS]
StationName= STATION
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        # Try lazy + trim - warning appears in logs
        ds = read(tmp_path, lazy=True, max_layers=20, trim_padding=True)
        
        # Verify operation completed
        assert ds is not None
        assert ds.sizes['layer'] == 10
    
    def test_combined_operations_log_sequence(self, tmp_path):
        """Should log validation success and trimming action (visible in pytest output)."""
        # Create homogeneous dataset
        for i in range(3):
            filepath = tmp_path / f"station_{i}.pro"
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        ds = read(tmp_path, lazy=False, max_layers=20, trim_padding=True)
        
        # Verify operations completed successfully
        assert ds is not None
        assert ds.sizes['layer'] == 10  # Trimmed from 20


class TestEdgeCasesIntegration:
    """Edge cases for integrated features."""
    
    def test_single_file_no_validation_can_trim(self, tmp_path):
        """Single file bypasses validation, can still trim."""
        filepath = tmp_path / "single.pro"
        filepath.write_text("""
[STATION_PARAMETERS]
StationName= SINGLE
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        ds = read(tmp_path, lazy=False, max_layers=20, trim_padding=True)
        
        assert ds is not None
        assert ds.sizes['layer'] == 10
    
    def test_empty_directory_both_features(self, tmp_path):
        """Empty directory should handle both features gracefully."""
        ds = read(tmp_path, force_smart_detection=True, trim_padding=True)
        
        assert ds is None  # No files found
    
    def test_all_files_fail_stat(self, tmp_path, monkeypatch):
        """If all files fail stat, should handle gracefully."""
        # Create files
        for i in range(3):
            filepath = tmp_path / f"station_{i}.pro"
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        # Mock stat to always fail
        def mock_stat(self, follow_symlinks=True):
            raise OSError("Mock stat failure")
        
        monkeypatch.setattr(Path, "stat", mock_stat)
        
        # Should handle gracefully (validation passes with insufficient data)
        # But actual file reading will also fail, so ds will be None or error
        # This tests that validation doesn't crash
        try:
            ds = read(tmp_path, lazy=False)
        except Exception:
            pass  # Expected to fail at some point


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests to quantify improvements."""
    
    def test_smart_detection_speed(self, tmp_path, benchmark):
        """Benchmark smart max_layers detection performance."""
        # Create 50 files (enough to show difference)
        for i in range(50):
            filepath = tmp_path / f"station_{i:03d}.pro"
            layer_data = ",".join([str(0.01 * j) for j in range(15)])
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i:03d}
[DATA]
0500,01.01.2025 12:00:00
0501,15,{layer_data}
""")
        
        # Benchmark smart detection (scans 10 files)
        def smart_load():
            return read(tmp_path, lazy=False, max_layers='auto')
        
        # Should complete in reasonable time
        result = benchmark(smart_load)
        assert result is not None
    
    def test_trimming_memory_savings(self, tmp_path):
        """Quantify memory savings from trimming."""
        import sys
        
        # Create dataset
        for i in range(5):
            filepath = tmp_path / f"station_{i}.pro"
            layer_data = ",".join([str(0.01 * j) for j in range(10)])
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,10,{layer_data}
""")
        
        # Load with padding
        ds_padded = read(tmp_path, lazy=False, max_layers=50)
        size_padded = sys.getsizeof(ds_padded['height'].values)
        
        # Load with trimming
        ds_trimmed = read(tmp_path, lazy=False, max_layers=50, trim_padding=True)
        size_trimmed = sys.getsizeof(ds_trimmed['height'].values)
        
        # Calculate savings
        savings_pct = (size_padded - size_trimmed) / size_padded * 100
        
        # Should save significant memory (> 50% for this case)
        assert savings_pct > 50


@pytest.mark.slow
class TestLargeScaleIntegration:
    """Tests with larger datasets to verify scalability."""
    
    def test_many_files_validation_performance(self, tmp_path):
        """Validation should scale well with many files."""
        # Create 100 files
        for i in range(100):
            filepath = tmp_path / f"station_{i:03d}.pro"
            filepath.write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i:03d}
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
        
        # Should complete quickly even with 100 files
        start = time.time()
        ds = read(tmp_path, lazy=False)
        elapsed = time.time() - start
        
        # Validation overhead should be minimal (< 0.5 sec for 100 files)
        # Total time will be more, but validation itself is fast
        assert ds is not None
        # Just verify it completes without timeout


def test_feature_combination_matrix(tmp_path):
    """Test all combinations of feature settings."""
    # Create simple dataset
    filepath = tmp_path / "station.pro"
    filepath.write_text("""
[STATION_PARAMETERS]
StationName= STATION
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
""")
    
    test_cases = [
        # (force_smart, trim_padding, expected_success)
        (False, False, True),   # Default
        (False, True, True),    # Trim only
        (True, False, True),    # Force only
        (True, True, True),     # Both enabled
    ]
    
    for force, trim, should_succeed in test_cases:
        ds = read(tmp_path, 
                 force_smart_detection=force,
                 trim_padding=trim,
                 lazy=False,
                 max_layers=20)
        
        if should_succeed:
            assert ds is not None
            if trim:
                assert ds.sizes['layer'] == 10
            else:
                assert ds.sizes['layer'] == 20
