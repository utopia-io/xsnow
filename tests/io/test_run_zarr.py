#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test runner for Zarr operations (no pytest required)

Run individual tests or all tests without pytest dependency.
"""

import sys
import traceback
from pathlib import Path
import tempfile
import shutil

# Import the test module
from test_zarr_operations import *


def run_test(test_func, *args):
    """Run a single test and report results."""
    test_name = test_func.__name__
    print(f"\n{'='*70}")
    print(f"Running: {test_name}")
    print(f"{'='*70}")
    
    try:
        test_func(*args)
        print(f"\n✅ PASSED: {test_name}")
        return True
    except AssertionError as e:
        print(f"\n❌ FAILED: {test_name}")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n💥 ERROR: {test_name}")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print(f"\n{'#'*70}")
    print("# ZARR OPERATIONS TEST SUITE (Simple Runner)")
    print(f"{'#'*70}\n")
    
    # Setup
    sample_dir = snp_gridded_dir()
    temp_dir = Path(tempfile.mkdtemp(prefix='zarr_test_'))
    helper_inst = ZarrTestHelper()
    
    print(f"Sample data: {sample_dir}")
    print(f"Temp dir: {temp_dir}")
    
    # Define tests
    tests = [
        # Basic reading
        (test_read_sample_data_eager, sample_dir),
        (test_read_sample_data_lazy, sample_dir),
        
        # Basic writing
        (test_write_zarr_basic, sample_dir, temp_dir, helper_inst),
        (test_write_zarr_with_int8, sample_dir, temp_dir, helper_inst),
        
        # Reading back
        (test_read_zarr_eager, sample_dir, temp_dir, helper_inst),
        (test_read_zarr_lazy, sample_dir, temp_dir, helper_inst),
        
        # Round-trip
        (test_roundtrip_eager, sample_dir, temp_dir, helper_inst),
        (test_roundtrip_lazy, sample_dir, temp_dir, helper_inst),
        
        # Chunking
        (test_adaptive_chunking_profile_data, sample_dir, helper_inst),
        (test_adaptive_chunking_timeseries_data, temp_dir, helper_inst),
        
        # Preprocessing
        (test_preprocess_slope_coordinates, sample_dir, helper_inst),
        (test_preprocess_drops_derived_coords, temp_dir, helper_inst),
        
        # Performance
        (test_compare_eager_vs_lazy_write, sample_dir, temp_dir, helper_inst),
    ]
    
    # Run tests
    results = []
    for test_func, *args in tests:
        passed = run_test(test_func, *args)
        results.append((test_func.__name__, passed))
    
    # Cleanup
    print(f"\n{'='*70}")
    print("Cleaning up...")
    print(f"{'='*70}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Removed: {temp_dir}")
    
    # Summary
    print(f"\n{'#'*70}")
    print("# TEST SUMMARY")
    print(f"{'#'*70}\n")
    
    passed = sum(1 for _, p in results if p)
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    
    print(f"\nDetailed results:")
    for name, passed_flag in results:
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"  {status}: {name}")
    
    # Exit code
    exit_code = 0 if failed == 0 else 1
    print(f"\nExit code: {exit_code}")
    return exit_code


if __name__ == '__main__':
    sys.exit(main())