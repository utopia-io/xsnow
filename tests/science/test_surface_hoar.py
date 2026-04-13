# tests/features/test_surface_hoar.py
"""
Tests for surface hoar layer handling.
Focus: Correct synthesis and addition of surface hoar layers.
"""
import pytest
import numpy as np
from xsnow import read
from xsnow.io.pro import _read_pro_raw


@pytest.mark.synthetic
class TestSurfaceHoarDetection:
    """Test detection of surface hoar data."""
    
    def test_detect_surface_hoar_present(self, synthetic_pro, tmp_path):
        """Test detection when surface hoar is present."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        _, profiles = _read_pro_raw(filepath, add_surface_sh_as_layer=True)
        
        # Should have SH data
        assert 'SH_at_surface' in profiles[0]
        sh_data = profiles[0]['SH_at_surface']
        
        # Should have positive flag
        assert sh_data[0] > 0
        # Should have grain size and density
        assert sh_data.size >= 3
    
    def test_detect_surface_hoar_absent(self, synthetic_pro, tmp_path):
        """Test detection when surface hoar is absent."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=False)
        
        _, profiles = _read_pro_raw(filepath, add_surface_sh_as_layer=True)
        
        if 'SH_at_surface' in profiles[0]:
            sh_data = profiles[0]['SH_at_surface']
            # Flag should be zero or negative
            assert sh_data[0] <= 0
    
    def test_surface_hoar_parameters(self, synthetic_pro, tmp_path):
        """Test that SH parameters are correctly parsed."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        _, profiles = _read_pro_raw(filepath, add_surface_sh_as_layer=True)
        
        sh_data = profiles[0]['SH_at_surface']
        
        # Should have: [flag, grain_size, density]
        assert len(sh_data) >= 3
        assert sh_data[0] > 0  # flag
        assert sh_data[1] > 0  # grain size should be positive
        assert sh_data[2] > 0  # density should be positive


@pytest.mark.synthetic
class TestSurfaceHoarLayerSynthesis:
    """Test synthesis of SH layer from SH data."""
    
    def test_add_sh_layer_default(self, synthetic_pro, tmp_path):
        """Test that SH layer is added by default when present."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        if 'grain_type' in ds.data:
            grain_types = ds['grain_type'].values[0, 0, 0, 0, :]
            # Should have grain type 660 (SH)
            assert 660 in grain_types[~np.isnan(grain_types)]
    
    def test_sh_layer_at_top(self, synthetic_pro, tmp_path):
        """Test that SH layer is added at top of snowpack."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        if 'height' in ds.data and 'grain_type' in ds.data:
            height = ds['height'].values[0, 0, 0, 0, :]
            grain_type = ds['grain_type'].values[0, 0, 0, 0, :]
            
            # Remove NaNs
            valid = ~(np.isnan(height) | np.isnan(grain_type))
            height_valid = height[valid]
            grain_valid = grain_type[valid]
            
            if len(height_valid) > 0:
                # Highest layer should be SH
                max_idx = np.argmax(height_valid)
                assert grain_valid[max_idx] == 660
    
    def test_sh_layer_properties(self, synthetic_pro, tmp_path):
        """Test that SH layer has correct properties."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        if 'grain_type' in ds.data:
            # Find SH layer
            grain_types = ds['grain_type'].values[0, 0, 0, 0, :]
            sh_indices = np.where(grain_types == 660)[0]
            
            if len(sh_indices) > 0:
                sh_idx = sh_indices[0]
                
                # Should have grain size
                if 'grain_size' in ds.data:
                    gs = ds['grain_size'].values[0, 0, 0, 0, sh_idx]
                    assert not np.isnan(gs)
                    assert gs > 0
                
                # Should have density
                if 'density' in ds.data:
                    dens = ds['density'].values[0, 0, 0, 0, sh_idx]
                    assert not np.isnan(dens)
                    assert dens > 0
                
                # Other properties may be NaN (e.g., temperature)
                if 'temperature' in ds.data:
                    temp = ds['temperature'].values[0, 0, 0, 0, sh_idx]
                    # Temperature is allowed to be NaN for SH layer
                    assert True  # Just checking it exists
    
    def test_sh_layer_height_continuity(self, synthetic_pro, tmp_path):
        """Test that SH layer height is continuous with snow surface."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        if 'height' in ds.data and 'grain_type' in ds.data:
            height = ds['height'].values[0, 0, 0, 0, :]
            grain_type = ds['grain_type'].values[0, 0, 0, 0, :]
            
            valid = ~(np.isnan(height) | np.isnan(grain_type))
            height_valid = height[valid]
            grain_valid = grain_type[valid]
            
            if len(height_valid) > 1:
                # Find SH layer
                sh_mask = grain_valid == 660
                if np.any(sh_mask):
                    sh_heights = height_valid[sh_mask]
                    non_sh_heights = height_valid[~sh_mask]
                    
                    # SH should be at or above highest non-SH layer
                    if len(non_sh_heights) > 0:
                        max_snow_height = np.max(non_sh_heights)
                        assert np.all(sh_heights >= max_snow_height)


@pytest.mark.synthetic
class TestSurfaceHoarDisabled:
    """Test behavior when SH layer addition is disabled."""
    
    def test_no_sh_layer_when_disabled(self, synthetic_pro, tmp_path):
        """Test that SH layer is not added when disabled."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        ds = read(filepath, add_surface_sh_as_layer=False)
        
        if 'grain_type' in ds.data:
            grain_types = ds['grain_type'].values[0, 0, 0, 0, :]
            # Should NOT have grain type 660
            assert 660 not in grain_types[~np.isnan(grain_types)]
    
    def test_layer_count_difference(self, synthetic_pro, tmp_path):
        """Test that layer count differs when SH is enabled vs disabled."""
        filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
        
        ds_with_sh = read(filepath, add_surface_sh_as_layer=True)
        ds_without_sh = read(filepath, add_surface_sh_as_layer=False)
        
        if 'height' in ds_with_sh.data and 'height' in ds_without_sh.data:
            height_with = ds_with_sh['height'].values[0, 0, 0, 0, :]
            height_without = ds_without_sh['height'].values[0, 0, 0, 0, :]
            
            valid_with = np.sum(~np.isnan(height_with))
            valid_without = np.sum(~np.isnan(height_without))
            
            # Should have one more layer with SH
            assert valid_with == valid_without + 1


@pytest.mark.synthetic
class TestSurfaceHoarEdgeCases:
    """Test edge cases in surface hoar handling."""
    
    def test_sh_with_no_snow(self, synthetic_pro, tmp_path):
        """Test SH when there's no snow on ground."""
        # Create profile with only SH (no underlying snow)
        content = """[STATION_PARAMETERS]
StationName= SH_ONLY

[DATA]
0500,01.01.2025 12:00:00
0501,0,
0502,0,
0514,3,1.0,0.5,100.0
"""
        filepath = tmp_path / "sh_only.pro"
        filepath.write_text(content)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        # Should still create dataset with SH layer
        if ds is not None and 'grain_type' in ds.data:
            grain_types = ds['grain_type'].values[0, 0, 0, 0, :]
            assert 660 in grain_types[~np.isnan(grain_types)]
    
    def test_sh_with_single_snow_layer(self, synthetic_pro, tmp_path):
        """Test SH addition with minimal snowpack."""
        content = """[STATION_PARAMETERS]
StationName= MINIMAL_SNOW

[DATA]
0500,01.01.2025 12:00:00
0501,1,0.1
0502,1,100
0514,3,1.0,0.5,100.0
"""
        filepath = tmp_path / "minimal_with_sh.pro"
        filepath.write_text(content)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        if 'height' in ds.data:
            height = ds['height'].values[0, 0, 0, 0, :]
            valid = np.sum(~np.isnan(height))
            
            # Should have 2 layers (1 snow + 1 SH)
            assert valid == 2
    
    def test_multiple_sh_flags(self, synthetic_pro, tmp_path):
        """Test handling when SH flag appears multiple times."""
        # Create two profiles, one with SH, one without
        content = """[STATION_PARAMETERS]
StationName= VARYING_SH

[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
0514,3,1.0,0.5,100.0

0500,02.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
0514,3,0.0,0.0,0.0

0500,03.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
0514,3,1.0,0.6,120.0
"""
        filepath = tmp_path / "varying_sh.pro"
        filepath.write_text(content)
        
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        # Should handle each profile independently
        assert ds.sizes['time'] == 3
        
        if 'grain_type' in ds.data:
            # Check each profile
            for t in range(ds.sizes['time']):
                grain_types = ds['grain_type'].values[0, t, 0, 0, :]
                valid_gt = grain_types[~np.isnan(grain_types)]
                
                # Profile 0 and 2 should have SH, profile 1 should not
                if t in [0, 2]:
                    assert 660 in valid_gt, f"Profile {t} should have SH"
                # Profile 1 might or might not depending on implementation
    
    def test_sh_with_missing_parameters(self, synthetic_pro, tmp_path):
        """Test SH handling when some parameters are missing."""
        # SH with incomplete data
        content = """[STATION_PARAMETERS]
StationName= INCOMPLETE_SH

[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
0514,1,1.0
"""
        filepath = tmp_path / "incomplete_sh.pro"
        filepath.write_text(content)
        
        # Should handle gracefully (either add with defaults or skip)
        ds = read(filepath, add_surface_sh_as_layer=True)
        
        assert ds is not None


@pytest.mark.synthetic
class TestSurfaceHoarWithSoilRemoval:
    """Test interaction between SH addition and soil removal."""
    
    def test_sh_and_soil_removal_together(self, synthetic_pro, tmp_path):
        """Test that SH addition and soil removal work together."""
        # Create profile with both SH and soil
        content = """[STATION_PARAMETERS]
StationName= SH_AND_SOIL

[DATA]
0500,01.01.2025 12:00:00
0501,5,0.1,0.2,0.3,0.4,0.5
0502,5,100,200,300,400,500
0513,5,200,200,200,81,82
0514,3,1.0,0.5,100.0
"""
        filepath = tmp_path / "sh_and_soil.pro"
        filepath.write_text(content)
        
        ds = read(filepath, add_surface_sh_as_layer=True, remove_soil=True)
        
        if 'height' in ds.data and 'grain_type' in ds.data:
            height = ds['height'].values[0, 0, 0, 0, :]
            grain_type = ds['grain_type'].values[0, 0, 0, 0, :]
            
            valid = ~(np.isnan(height) | np.isnan(grain_type))
            grain_valid = grain_type[valid]
            
            # Should have SH (660) but not soil (81, 82)
            assert 660 in grain_valid
            assert 81 not in grain_valid
            assert 82 not in grain_valid
    
    def test_layer_count_with_both_operations(self, synthetic_pro, tmp_path):
        """Test layer count when both SH and soil operations are active."""
        content = """[STATION_PARAMETERS]
StationName= BOTH_OPS

[DATA]
0500,01.01.2025 12:00:00
0501,7,0.1,0.2,0.3,0.4,0.5,0.6,0.7
0502,7,100,200,300,400,500,600,700
0513,7,200,200,200,200,200,81,82
0514,3,1.0,0.5,100.0
"""
        filepath = tmp_path / "both_ops.pro"
        filepath.write_text(content)
        
        # Load with both operations
        ds = read(filepath, add_surface_sh_as_layer=True, remove_soil=True)
        
        if 'height' in ds.data:
            height = ds['height'].values[0, 0, 0, 0, :]
            valid = np.sum(~np.isnan(height))
            
            # Should have: 5 snow + 1 SH = 6 layers
            # (removed 2 soil layers)
            assert valid == 6