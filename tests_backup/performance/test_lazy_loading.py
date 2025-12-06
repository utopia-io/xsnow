# tests/performance/test_lazy_loading.py
"""
Test lazy loading using ACTUAL xsnow implementation.

Tests:
- xsnow.read() performance
- xsnowDataset lazy loading (if it has lazy_threshold)
- Memory efficiency

Run:
    pytest tests/performance/test_lazy_loading.py -v -s
"""

import pytest
from pathlib import Path
from .profiling_utils import Benchmark, track_memory, Timer
import gc

# Import actual xsnow modules - CUSTOMIZED FOR YOUR XSNOW
try:
    from xsnow import read
    HAS_READ = True
except ImportError:
    HAS_READ = False
    print("⚠️  xsnow.read() not available")

try:
    from xsnow import xsnowDataset
    HAS_DATASET = True
except ImportError:
    HAS_DATASET = False
    print("⚠️  xsnowDataset not available")

try:
    from xsnow import xsnow_io
    HAS_IO = True
except ImportError:
    HAS_IO = False
    print("⚠️  xsnow_io module not available")


# Skip all tests if xsnow not available
pytestmark = pytest.mark.skipif(
    not (HAS_READ or HAS_DATASET or HAS_IO),
    reason="xsnow modules not available for testing"
)


# ============================================================================
# TEST FIXTURES - Real .pro files
# ============================================================================

@pytest.fixture(scope="session")
def small_real_dataset(tmp_path_factory):
    """Small: 5 files × 20 profiles each."""
    data_dir = tmp_path_factory.mktemp("real_small")
    
    for file_idx in range(5):
        profiles = []
        for profile_idx in range(20):
            profiles.extend([
                f"0500,{profile_idx+1:02d}.01.2025 12:00:00",
                f"0501,10," + ",".join([f"{i*0.1:.1f}" for i in range(10)]),
                f"0502,10," + ",".join([f"{i*100}" for i in range(10)])
            ])
        
        (data_dir / f"station_{file_idx:03d}.pro").write_text(
            f"[STATION_PARAMETERS]\nStationName=STATION_{file_idx:03d}\n\n[DATA]\n" + 
            "\n".join(profiles)
        )
    
    return data_dir


@pytest.fixture(scope="session")
def large_real_dataset(tmp_path_factory):
    """Large: 50 files × 100 profiles each."""
    data_dir = tmp_path_factory.mktemp("real_large")
    
    for file_idx in range(50):
        profiles = []
        for profile_idx in range(100):
            profiles.extend([
                f"0500,{profile_idx+1:02d}.01.2025 12:00:00",
                f"0501,20," + ",".join([f"{i*0.1:.1f}" for i in range(20)]),
                f"0502,20," + ",".join([f"{i*10}" for i in range(20)])
            ])
        
        (data_dir / f"station_{file_idx:03d}.pro").write_text(
            f"[STATION_PARAMETERS]\nStationName=STATION_{file_idx:03d}\n\n[DATA]\n" + 
            "\n".join(profiles)
        )
    
    return data_dir


# ============================================================================
# TEST: xsnow.read() function
# ============================================================================

@pytest.mark.skipif(not HAS_READ, reason="xsnow.read() not available")
class TestXsnowRead:
    """Test the xsnow.read() function."""
    
    def test_read_small_dataset(self, small_real_dataset):
        """Test reading small dataset with xsnow.read()."""
        print("\n" + "="*70)
        print("XSNOW.READ() - Small Dataset (5 files)")
        print("="*70)
        
        with Timer() as t:
            data = read(small_real_dataset)
        
        print(f"\nRead time: {t.elapsed*1000:.2f}ms")
        print(f"Data type: {type(data).__name__}")
        
        # Try to get length
        try:
            length = len(data)
            print(f"Number of items: {length}")
        except:
            print("Data is not directly indexable")
        
        print("="*70)
    
    def test_read_large_dataset(self, large_real_dataset):
        """Test reading large dataset with xsnow.read()."""
        print("\n" + "="*70)
        print("XSNOW.READ() - Large Dataset (50 files)")
        print("="*70)
        
        with Timer() as t:
            data = read(large_real_dataset)
        
        print(f"\nRead time: {t.elapsed*1000:.2f}ms")
        print(f"Data type: {type(data).__name__}")
        
        print("="*70)
    
    def test_read_timing_comparison(self, small_real_dataset, large_real_dataset):
        """Compare read times for different dataset sizes."""
        print("\n" + "="*70)
        print("XSNOW.READ() - Timing Comparison")
        print("="*70)
        
        # Small dataset
        with Timer() as t1:
            data_small = read(small_real_dataset)
        time_small = t1.elapsed * 1000
        
        # Large dataset  
        with Timer() as t2:
            data_large = read(large_real_dataset)
        time_large = t2.elapsed * 1000
        
        ratio = time_large / time_small if time_small > 0 else 0
        
        print(f"\nSmall (5 files):  {time_small:>8.2f}ms")
        print(f"Large (50 files): {time_large:>8.2f}ms")
        print(f"Ratio:            {ratio:>8.2f}x")
        
        if ratio < 5:
            print("\n✅ Excellent scaling (sublinear)")
        elif ratio < 10:
            print("\n⚠️  Linear scaling (expected)")
        else:
            print("\n❌ Superlinear scaling (check for issues)")
        
        print("="*70)
    
    def test_read_memory(self, large_real_dataset):
        """Test memory usage of xsnow.read()."""
        print("\n" + "="*70)
        print("XSNOW.READ() - Memory Usage")
        print("="*70)
        
        gc.collect()
        
        with track_memory() as t:
            data = read(large_real_dataset)
        
        mem_used = t.get_increase()
        
        print(f"\nMemory used: {mem_used:.2f} MB")
        print(f"Files: 50")
        print(f"Avg per file: {mem_used/50:.2f} MB")
        
        if mem_used < 10:
            print("\n✅ Excellent memory efficiency")
        elif mem_used < 50:
            print("\n⚠️  Moderate memory usage")
        else:
            print("\n❌ High memory usage")
        
        print("="*70)


# ============================================================================
# TEST: xsnowDataset class
# ============================================================================

@pytest.mark.skipif(not HAS_DATASET, reason="xsnowDataset not available")
class TestXsnowDataset:
    """Test xsnowDataset class."""
    
    def test_dataset_basic(self, small_real_dataset):
        """Test basic xsnowDataset usage."""
        print("\n" + "="*70)
        print("XSNOWDATASET - Basic Usage")
        print("="*70)
        
        with Timer() as t:
            # Try different ways to create dataset
            try:
                ds = xsnowDataset(small_real_dataset)
            except TypeError as e:
                print(f"⚠️  xsnowDataset() failed: {e}")
                print("   Trying with read() instead...")
                ds = read(small_real_dataset)
        
        print(f"\nInit time: {t.elapsed*1000:.2f}ms")
        print(f"Type: {type(ds).__name__}")
        
        # Check for lazy loading attributes
        if hasattr(ds, '_lazy'):
            print(f"Has _lazy: {ds._lazy}")
        if hasattr(ds, 'lazy_threshold'):
            print(f"lazy_threshold: {ds.lazy_threshold}")
        if hasattr(ds, '_cache'):
            print("Has _cache: True (lazy loading)")
        
        print("="*70)
    
    def test_dataset_with_lazy_threshold(self, large_real_dataset):
        """Test if xsnowDataset supports lazy_threshold parameter."""
        print("\n" + "="*70)
        print("XSNOWDATASET - Lazy Threshold Test")
        print("="*70)
        
        # Try to create with lazy_threshold
        try:
            import inspect
            sig = inspect.signature(xsnowDataset.__init__)
            params = sig.parameters
            has_lazy_param = 'lazy_threshold' in params or 'lazy' in params
            
            print(f"\nxsnowDataset signature: {sig}")
            print(f"Has lazy parameter: {has_lazy_param}")
            
            if has_lazy_param:
                print("\n✅ xsnowDataset supports lazy loading!")
                
                # Test with different thresholds
                with Timer() as t1:
                    ds_eager = xsnowDataset(large_real_dataset, lazy_threshold=100)
                
                with Timer() as t2:
                    ds_lazy = xsnowDataset(large_real_dataset, lazy_threshold=5)
                
                print(f"\nEager (threshold=100): {t1.elapsed*1000:.2f}ms")
                print(f"Lazy (threshold=5):    {t2.elapsed*1000:.2f}ms")
                
                if t2.elapsed < t1.elapsed * 0.5:
                    print("\n✅ Lazy loading working well!")
                else:
                    print("\n⚠️  Lazy mode not significantly faster")
            else:
                print("\n⚠️  xsnowDataset does not have lazy_threshold parameter")
                print("   Lazy loading may be automatic or not implemented")
                
        except Exception as e:
            print(f"\n⚠️  Could not inspect xsnowDataset: {e}")
        
        print("="*70)


# ============================================================================
# COMPREHENSIVE TEST
# ============================================================================

class TestComprehensiveXsnow:
    """Comprehensive test of xsnow performance."""
    
    def test_comprehensive(self, small_real_dataset, large_real_dataset):
        """Run comprehensive performance test."""
        print("\n" + "="*70)
        print("COMPREHENSIVE XSNOW PERFORMANCE TEST")
        print("="*70)
        
        results = {}
        
        # Test 1: xsnow.read() on small dataset
        if HAS_READ:
            print("\n📊 Test 1: xsnow.read() - Small Dataset")
            with Timer() as t:
                data_small = read(small_real_dataset)
            results['read_small'] = t.elapsed * 1000
            print(f"   Time: {results['read_small']:.2f}ms")
        
        # Test 2: xsnow.read() on large dataset
        if HAS_READ:
            print("\n📊 Test 2: xsnow.read() - Large Dataset")
            with Timer() as t:
                data_large = read(large_real_dataset)
            results['read_large'] = t.elapsed * 1000
            print(f"   Time: {results['read_large']:.2f}ms")
            
            if 'read_small' in results:
                ratio = results['read_large'] / results['read_small']
                print(f"   Ratio: {ratio:.2f}x")
                
                if ratio < 5:
                    print("   ✅ Excellent scaling")
                elif ratio < 10:
                    print("   ⚠️  Linear scaling")
                else:
                    print("   ❌ Poor scaling")
        
        # Test 3: Memory usage
        if HAS_READ:
            print("\n📊 Test 3: Memory Usage")
            gc.collect()
            
            with track_memory() as t:
                data = read(large_real_dataset)
            
            results['memory'] = t.get_increase()
            print(f"   Memory: {results['memory']:.2f}MB")
            
            if results['memory'] < 10:
                print("   ✅ Excellent")
            elif results['memory'] < 50:
                print("   ⚠️  Moderate")
            else:
                print("   ❌ High")
        
        # Final recommendations
        print("\n" + "="*70)
        print("🎯 RECOMMENDATIONS")
        print("="*70)
        
        if 'read_large' in results and 'read_small' in results:
            ratio = results['read_large'] / results['read_small']
            if ratio < 5:
                print("\n✅ xsnow.read() has good performance characteristics")
                print(f"   • Sublinear scaling ({ratio:.1f}x for 10x more files)")
            else:
                print("\n💡 xsnow.read() performance is linear")
                print(f"   • {ratio:.1f}x slower for 10x more files")
                print("   • May benefit from lazy loading optimization")
        
        if 'memory' in results:
            if results['memory'] < 10:
                print("\n✅ Memory usage is excellent")
            else:
                print(f"\n💡 Memory usage is {results['memory']:.0f}MB for 50 files")
                print("   • Consider lazy loading for large datasets")
        
        print("\n" + "="*70)


# ============================================================================
# HELPER TEST - Print xsnow API info
# ============================================================================

def test_print_xsnow_info():
    """Print xsnow API information."""
    print("\n" + "="*70)
    print("XSNOW API INFORMATION")
    print("="*70)
    
    print("\n📦 Available modules:")
    print(f"   xsnow.read():       {HAS_READ}")
    print(f"   xsnowDataset:       {HAS_DATASET}")
    print(f"   xsnow_io:           {HAS_IO}")
    
    if HAS_READ:
        print("\n✅ xsnow.read() signature:")
        try:
            import inspect
            sig = inspect.signature(read)
            print(f"   {sig}")
        except Exception as e:
            print(f"   Could not get signature: {e}")
    
    if HAS_DATASET:
        print("\n✅ xsnowDataset signature:")
        try:
            import inspect
            sig = inspect.signature(xsnowDataset.__init__)
            print(f"   {sig}")
            
            # Check for lazy-related parameters
            params = list(sig.parameters.keys())
            lazy_params = [p for p in params if 'lazy' in p.lower()]
            if lazy_params:
                print(f"   Lazy parameters: {lazy_params}")
            else:
                print("   No obvious lazy loading parameters")
        except Exception as e:
            print(f"   Could not get signature: {e}")
    
    print("\n" + "="*70)