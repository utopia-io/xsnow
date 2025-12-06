# tests/test_quick_check.py
"""Quick sanity check for string coordinates."""
import pytest
from xsnow import read

@pytest.mark.synthetic
def test_string_coords_work(synthetic_pro, tmp_path):
    """Minimal test - if this passes, basic functionality works."""
    # Create file
    filepath = synthetic_pro.basic(tmp_path, station_name='QUICK_TEST', n_layers=3)
    
    # Read
    ds = read(filepath)
    
    print(ds.data_vars)  # What variables exist?
    print(list(ds.coords))  # What coordinates exist?
    
    # Verify
    assert ds is not None
    assert ds.data.location.dtype.kind in ('U', 'S', 'O')
    assert str(ds.data.location.values[0]) == 'QUICK_TEST'
    assert 'mapping' not in ds.data.location.attrs
    
    print("✅ String coordinates working!")

if __name__ == '__main__':
    
    
    
    
    pytest.main([__file__, '-v'])