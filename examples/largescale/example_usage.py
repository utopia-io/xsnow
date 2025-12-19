#!/usr/bin/env python3
"""
Example usage of SnowpackUpdater

Demonstrates:
- Auto-detection (recommended)
- Explicit mode control
- Querying results
"""

from snowpack_updater import SnowpackUpdater
from pathlib import Path


def example_auto_detect():
    """Example: Auto-detect mode (recommended)"""
    print("="*70)
    print("Example 1: Auto-Detect Mode (Recommended)")
    print("="*70)
    
    updater = SnowpackUpdater(
        data_dir='/home/snowpack/runs/output.2024',
        archive_file='snowpack_operational.nc',
        window_days=90,
        lazy=True,
        n_cpus=15
    )
    
    print(f"\nDetected mode: {updater.mode}")
    print(f"Window days: {updater.window_days}")
    
    # Run update
    ds = updater.run()
    
    # Query results
    if ds:
        print(f"\n📊 Results:")
        print(f"   Locations: {len(ds.data.location):,}")
        print(f"   Time points: {len(ds.data.time):,}")
        print(f"   Date range: {ds.data.time.values[0]} to {ds.data.time.values[-1]}")
        
        if 'max_HS' in ds.data:
            max_hs = float(ds.data.max_HS.max().values)
            print(f"   Max HS: {max_hs:.2f} m")


def example_initial_load():
    """Example: Force initial load to create full historical archive"""
    print("\n" + "="*70)
    print("Example 2: Force Initial Load (Full History)")
    print("="*70)
    
    updater = SnowpackUpdater(
        data_dir='/home/snowpack/runs/output.2024',
        archive_file='snowpack_full_history.nc',
        window_days=90,  # Ignored in initial mode
        mode='initial',   # Force initial
        lazy=True
    )
    
    print(f"\nMode: {updater.mode}")
    print(f"Window days: {updater.window_days} (0 = keep all history)")
    
    ds = updater.run()
    
    # This creates a full historical archive
    # All timesteps from file creation to now!


def example_update_only():
    """Example: Force update mode (requires existing archive)"""
    print("\n" + "="*70)
    print("Example 3: Force Update Mode")
    print("="*70)
    
    # Check if archive exists
    archive_path = Path('snowpack_operational.nc')
    
    if not archive_path.exists():
        print(f"\n⚠️  Archive not found: {archive_path}")
        print(f"   Skipping update-only example")
        print(f"   Run example_auto_detect() first to create archive")
        return
    
    updater = SnowpackUpdater(
        data_dir='/home/snowpack/runs/output.2024',
        archive_file='snowpack_operational.nc',
        window_days=90,
        mode='update',  # Force update, will error if no archive
        lazy=True
    )
    
    print(f"\nMode: {updater.mode}")
    print(f"Window days: {updater.window_days}")
    
    ds = updater.run()


def example_multiple_archives():
    """Example: Maintain multiple archives with different windows"""
    print("\n" + "="*70)
    print("Example 4: Multiple Archives")
    print("="*70)
    
    archives = [
        ('quick.nc', 30, 'Quick updates'),
        ('operational.nc', 90, 'Main forecasting'),
        ('extended.nc', 180, 'Seasonal analysis'),
    ]
    
    for archive_file, window_days, description in archives:
        print(f"\n📦 Creating {description} archive ({window_days} days)...")
        
        updater = SnowpackUpdater(
            data_dir='/home/snowpack/runs/output.2024',
            archive_file=archive_file,
            window_days=window_days,
            lazy=True
        )
        
        print(f"   Mode: {updater.mode}")
        print(f"   Window: {updater.window_days} days")
        
        # In production, you'd run this:
        # ds = updater.run()
        
        print(f"   (Skipped actual run in example)")


def example_query_archive():
    """Example: Query an existing archive"""
    print("\n" + "="*70)
    print("Example 5: Query Existing Archive")
    print("="*70)
    
    archive_path = Path('snowpack_operational.nc')
    
    if not archive_path.exists():
        print(f"\n⚠️  Archive not found: {archive_path}")
        print(f"   Create it first by running example_auto_detect()")
        return
    
    import xsnow
    
    print(f"\n📖 Reading archive: {archive_path}")
    ds = xsnow.read(archive_path)
    
    print(f"\n📊 Archive contents:")
    print(f"   Locations: {len(ds.data.location):,}")
    print(f"   Time points: {len(ds.data.time):,}")
    print(f"   Variables: {list(ds.data.data_vars.keys())}")
    
    if 'time' in ds.data.dims:
        print(f"   Date range: {ds.data.time.values[0]} to {ds.data.time.values[-1]}")
    
    if 'HS' in ds.data:
        print(f"\n❄️  Snow height statistics:")
        hs_mean = float(ds.data.HS.mean().values)
        hs_max = float(ds.data.HS.max().values)
        print(f"   Mean HS: {hs_mean:.2f} m")
        print(f"   Max HS: {hs_max:.2f} m")
    
    if 'max_HS' in ds.data:
        max_hs_overall = float(ds.data.max_HS.max().values)
        print(f"   Max HS (all time): {max_hs_overall:.2f} m")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("SNOWPACK UPDATER - USAGE EXAMPLES")
    print("="*70)
    
    # Note: In real usage, you'd only run one at a time!
    # These examples show different use cases
    
    print("\n📝 These examples demonstrate different usage patterns.")
    print("   In production, you typically just run example 1 (auto-detect).")
    print("   Uncomment the example you want to run:\n")
    
    # Uncomment the example you want to run:
    
    # Example 1: Auto-detect (recommended for production)
    # example_auto_detect()
    
    # Example 2: Force initial load (create full historical archive)
    # example_initial_load()
    
    # Example 3: Force update (require existing archive)
    # example_update_only()
    
    # Example 4: Multiple archives with different windows
    # example_multiple_archives()
    
    # Example 5: Query existing archive
    # example_query_archive()
    
    print("\n💡 To run an example, uncomment it in the main() function")
    print("   and run: python example_usage.py\n")


if __name__ == '__main__':
    main()
