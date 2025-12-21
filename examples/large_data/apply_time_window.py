#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Time Window Applicator

If you have an existing archive, this is BY FAR the fastest way to update it.
Just applies a time window - no file reloading needed.

Usage:
    python apply_time_window.py snowpack_operational.nc --days 14
"""

import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import argparse


def apply_time_window(
    archive_file: str,
    window_days: int,
    output_file: str = None
):
    """
    Apply time window to existing archive.
    
    This is MUCH faster than reloading 114,300 files!
    """
    
    archive_path = Path(archive_file)
    
    if not archive_path.exists():
        print(f"❌ Archive not found: {archive_path}")
        print(f"   You need to create it first")
        return False
    
    # Default output
    if output_file is None:
        output_file = archive_path.stem + f'_{window_days}day.nc'
    
    output_path = Path(output_file)
    
    print("="*70)
    print("TIME WINDOW APPLICATOR")
    print("="*70)
    print(f"Input:  {archive_path}")
    print(f"Output: {output_path}")
    print(f"Window: {window_days} days")
    print()
    
    # Load archive
    print("Loading archive...")
    ds = xr.open_dataset(archive_path)
    
    print(f"  Locations: {len(ds.location):,}")
    print(f"  Time points: {len(ds.time):,}")
    print(f"  Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    file_size_mb = archive_path.stat().st_size / (1024**2)
    print(f"  Size: {file_size_mb:.1f} MB")
    
    # Apply window
    print()
    print(f"Applying {window_days}-day window...")
    
    cutoff = datetime.now() - timedelta(days=window_days)
    cutoff_str = cutoff.isoformat()
    
    print(f"  Cutoff date: {cutoff_str[:10]}")
    
    ds_windowed = ds.sel(time=slice(cutoff_str, None))
    
    n_removed = len(ds.time) - len(ds_windowed.time)
    n_kept = len(ds_windowed.time)
    
    print(f"  Removed: {n_removed:,} old timesteps")
    print(f"  Kept: {n_kept:,} timesteps")
    
    # Save
    print()
    print("Saving windowed archive...")
    
    ds_windowed.to_netcdf(output_path)
    
    output_size_mb = output_path.stat().st_size / (1024**2)
    savings_mb = file_size_mb - output_size_mb
    savings_pct = (savings_mb / file_size_mb) * 100
    
    print(f"  ✅ Saved: {output_size_mb:.1f} MB")
    print(f"  Savings: {savings_mb:.1f} MB ({savings_pct:.1f}% smaller)")
    
    # Close
    ds.close()
    ds_windowed.close()
    
    print()
    print("="*70)
    print("✅ COMPLETE")
    print("="*70)
    print(f"Windowed archive: {output_path}")
    print()
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Apply time window to existing SNOWPACK archive",
        epilog="""
Examples:
  # Keep last 14 days
  python apply_time_window.py snowpack_operational.nc --days 14
  
  # Keep last 30 days, specify output
  python apply_time_window.py snowpack_operational.nc --days 30 --output recent.nc
  
  # Keep last 90 days
  python apply_time_window.py snowpack_operational.nc --days 90
        """
    )
    
    parser.add_argument(
        'archive',
        help='Existing archive file'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=14,
        help='Days to keep (default: 14)'
    )
    parser.add_argument(
        '--output',
        help='Output file (default: <archive>_<days>day.nc)'
    )
    
    args = parser.parse_args()
    
    apply_time_window(
        archive_file=args.archive,
        window_days=args.days,
        output_file=args.output
    )
