#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNOWPACK Operational Archive - Current State Approach

Key insight: Operational archive = CURRENT STATE, not history!

Instead of creating new batches every run (4x/day = tons of redundancy),
maintain ONE operational archive that gets updated with latest data.

Perfect for 4 runs/day scenario:
- 6 AM: Update operational.nc with latest
- 12 PM: Update operational.nc with latest (replaces old timestamps)
- 6 PM: Update operational.nc with latest
- 12 AM: Update operational.nc with latest

Result: ONE file, always current, no redundancy!
"""

import xsnow
import xarray as xr
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import gc
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OperationalArchive:
    """
    Maintains current operational state.
    
    Updates existing archive rather than creating new batches.
    Perfect for frequent runs (4x/day).
    """
    
    def __init__(
        self,
        data_dir: str,
        archive_file: str = 'operational.nc',
        tracking_file: str = 'file_tracking.json',
        window_days: int = 14,
        batch_size: int = 20000,  # LARGER batches - processing ALL files anyway!
        n_cpus: int = 3,
        modification_threshold: int = 60
    ):
        """
        Operational archive for SNOWPACK that updates ALL files every run.
        
        Key insight: SNOWPACK appends new timestamp to ALL 114,300 files each run.
        So we always process all files, not just "changed" files.
        
        Strategy:
        - Use large batches (20,000 files) for efficiency
        - Replace existing operational.nc with new data
        - Keep only last N days (rolling window)
        """
        self.data_dir = Path(data_dir)
        self.archive_file = Path(archive_file)
        self.tracking_file = Path(tracking_file)
        self.window_days = window_days
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.modification_threshold = modification_threshold
        
        self.file_tracking = {}
        self.failed_files = set()  # Track files that failed to process
        self.temp_dir = Path('temp_operational')
    
    def update(self):
        """
        Update operational archive with latest SNOWPACK data.
        
        SNOWPACK updates ALL 114,300 files every run (appends new timestep).
        So this will process all files, not just a subset.
        
        Time estimate: 1-2 hours for full update
        """
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("OPERATIONAL ARCHIVE UPDATE - CURRENT STATE")
        logger.info("="*70)
        logger.info(f"Archive: {self.archive_file}")
        logger.info(f"Window: {self.window_days} days (rolling)")
        logger.info("")
        logger.info("Note: SNOWPACK updates ALL files each run")
        logger.info("      This will process all 114,300 files")
        logger.info("")
        
        # Load tracking
        self._load_tracking()
        
        # Find changed files
        changed_files = self._find_changed_files()
        
        if not changed_files:
            logger.info("✅ No changes - archive is current")
            
            # Still apply time window to remove old data
            if self.archive_file.exists():
                self._apply_time_window()
            
            self._show_status()
            return
        
        logger.info(f"Found {len(changed_files):,} changed files")
        
        # Sanity check
        total_files = len(list(self.data_dir.rglob('*.pro'))) + len(list(self.data_dir.rglob('*.smet')))
        pct_changed = (len(changed_files) / total_files * 100) if total_files > 0 else 0
        
        if pct_changed > 90:
            logger.info(f"  ({pct_changed:.0f}% of all files - SNOWPACK updated everything)")
        
        logger.info("")
        
        # Create temp batches with new data
        logger.info("Processing changed data...")
        new_batches = self._create_temp_batches(changed_files)
        
        if not new_batches:
            logger.warning("No new data processed")
            return
        
        # Merge with existing archive
        self._merge_with_archive(new_batches)
        
        # Apply time window
        self._apply_time_window()
        
        # Clean up
        self._cleanup()
        
        # Update tracking
        self._update_tracking()
        
        total_time = time.time() - start_time
        
        logger.info("")
        logger.info("="*70)
        logger.info("✅ ARCHIVE UPDATED")
        logger.info("="*70)
        logger.info(f"Time: {total_time/60:.1f} minutes")
        
        self._show_status()
    
    def _load_tracking(self):
        """Load file modification tracking."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                self.file_tracking = json.load(f)
            logger.info(f"Loaded tracking: {len(self.file_tracking):,} files")
        else:
            logger.info("Initializing tracking...")
            self.file_tracking = {}
            all_files = list(self.data_dir.rglob('*.pro')) + list(self.data_dir.rglob('*.smet'))
            for filepath in all_files:
                file_key = str(filepath.relative_to(self.data_dir))
                self.file_tracking[file_key] = 0
            logger.info(f"  Initialized: {len(self.file_tracking):,} files")
    
    def _find_changed_files(self):
        """
        Find files that changed since last run.
        
        Uses modification_threshold as first-pass filter for efficiency,
        but actual lock errors are caught during file reading with try-except.
        """
        logger.info("Scanning for changes...")
        
        all_files = list(self.data_dir.rglob('*.pro')) + list(self.data_dir.rglob('*.smet'))
        current_time = time.time()
        
        changed = []
        skipped_recent = 0
        
        for filepath in all_files:
            try:
                file_key = str(filepath.relative_to(self.data_dir))
                
                try:
                    mtime = filepath.stat().st_mtime
                except (OSError, PermissionError):
                    # Can't access file - skip for now
                    continue
                
                # Skip very recent files (optimization - likely still being written)
                # This is a first-pass filter, actual locks caught during read
                if current_time - mtime < self.modification_threshold:
                    skipped_recent += 1
                    continue
                
                # Check if changed
                tracked_mtime = self.file_tracking.get(file_key, 0)
                if mtime > tracked_mtime:
                    changed.append(filepath)
                    
            except Exception:
                continue
        
        if skipped_recent > 0:
            logger.info(f"  Skipped {skipped_recent:,} recent files (<{self.modification_threshold}s old)")
        
        return changed
    
    def _create_temp_batches(self, changed_files):
        """
        Create temporary batches with new data.
        
        Uses try-except to handle files still locked by SNOWPACK.
        Failed files are tracked and can be retried.
        """
        self.temp_dir.mkdir(exist_ok=True)
        
        n_batches = (len(changed_files) + self.batch_size - 1) // self.batch_size
        batch_files = []
        failed_files = []  # Track files that failed to read
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(changed_files))
            batch = changed_files[start_idx:end_idx]
            
            logger.info(f"  Batch {i+1}/{n_batches}: {len(batch):,} files")
            
            # Track which files actually succeed
            successful_files = []
            
            try:
                # EAGER loading - load data immediately, much faster for batch processing
                ds = xsnow.read(
                    batch,
                    lazy=False,  # ✅ Load immediately, no Dask overhead
                    n_cpus_use=min(self.n_cpus, 3)  # ✅ Limit workers to avoid semaphore leaks
                )
                
                if ds is None:
                    logger.warning("  ⚠️  No data returned from xsnow.read()")
                    # All files in batch failed
                    failed_files.extend(batch)
                    continue
                
                # Apply window to new data
                if self.window_days > 0:
                    cutoff = datetime.now() - timedelta(days=self.window_days)
                    ds = ds.sel(time=slice(cutoff.isoformat(), None))
                
                # CRITICAL CHECK: Verify batch has layer dimension
                if 'layer' not in ds.dims:
                    logger.warning(f"    ⚠️  Batch missing 'layer' dimension - incompatible structure")
                    logger.warning(f"    This batch has no snowpack layers (likely flat/bare terrain)")
                    logger.warning(f"    Skipping these {len(batch):,} files")
                    failed_files.extend(batch)
                    del ds
                    gc.collect()
                    continue
                
                batch_file = self.temp_dir / f'new_batch_{i:04d}.nc'
                
                # Save batch file
                try:
                    ds.to_netcdf(batch_file)
                    size_mb = batch_file.stat().st_size / (1024**2)
                    
                    # Validate batch - detect suspiciously small files
                    if size_mb < 10 and len(batch) > 100:
                        logger.warning(f"    ⚠️  Batch is suspiciously small ({size_mb:.1f} MB for {len(batch)} files)")
                        logger.warning(f"    This may indicate corrupted or empty data")
                        logger.warning(f"    Skipping this batch - files will be retried on next run")
                        batch_file.unlink()  # Delete bad batch
                        failed_files.extend(batch)
                        continue
                    
                    logger.info(f"    ✅ {size_mb:.1f} MB (layer dim: {ds.dims['layer']})")
                    batch_files.append(batch_file)
                    
                except (PermissionError, OSError) as e:
                    logger.warning(f"    ⚠️  Cannot save batch file: {e}")
                    logger.info(f"    Will retry these {len(batch):,} files on next run")
                    failed_files.extend(batch)
                    continue
                
                # Mark these files as successful
                successful_files = batch
                
                # Explicit cleanup to avoid semaphore leaks
                del ds
                gc.collect()
                
                # Force cleanup of multiprocessing resources
                import multiprocessing
                multiprocessing.resource_tracker.ensure_running()
                
            except PermissionError as e:
                # Files locked by SNOWPACK - expected during concurrent operation
                logger.warning(f"    ⚠️  Permission error (files locked): {e}")
                logger.info(f"    Will retry these {len(batch):,} files on next run")
                failed_files.extend(batch)
                
            except OSError as e:
                # I/O errors - file busy, etc.
                if "Resource busy" in str(e) or "locked" in str(e).lower():
                    logger.warning(f"    ⚠️  Files busy: {e}")
                    logger.info(f"    Will retry these {len(batch):,} files on next run")
                    failed_files.extend(batch)
                else:
                    logger.error(f"    ❌ I/O error: {e}")
                    failed_files.extend(batch)
                    
            except Exception as e:
                logger.error(f"    ❌ Failed: {e}")
                # Don't know which specific files failed, so retry all
                failed_files.extend(batch)
        
        # Report on failed files
        if failed_files:
            logger.warning("")
            logger.warning(f"⚠️  {len(failed_files):,} files could not be processed")
            logger.warning(f"   (likely still locked by SNOWPACK)")
            logger.warning(f"   These files will be retried on next run")
            logger.warning("")
            
            # Don't update tracking for failed files - they'll be detected as changed next time
            self._save_failed_files(failed_files)
        
        return batch_files
    
    def _save_failed_files(self, failed_files):
        """
        Save list of failed files so we don't update their tracking.
        They'll be detected as changed on next run.
        """
        # Store in instance variable for use in _update_tracking
        self.failed_files = set(str(f.relative_to(self.data_dir)) for f in failed_files)
    
    def _merge_with_archive(self, new_batches):
        """Merge new data with existing archive using hierarchical merging."""
        logger.info("")
        logger.info("Merging with existing archive...")
        
        # Ensure parent directory exists
        self.archive_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.archive_file.exists():
            # No existing archive - create new
            logger.info("  No existing archive - creating new")
            
            if len(new_batches) == 1:
                # Just one batch - copy it
                logger.info("  Single batch - copying directly")
                shutil.copy(new_batches[0], self.archive_file)
                
            elif len(new_batches) <= 5:
                # Few batches - merge directly
                logger.info(f"  Merging {len(new_batches)} batches directly...")
                ds_new = xr.open_mfdataset(
                    [str(f) for f in new_batches],
                    combine='nested',
                    concat_dim='location',
                    parallel=False,
                    engine='netcdf4',
                    join='outer',  # Allow different dimension sizes
                    preprocess=self._preprocess_for_merge  # Normalize datasets
                )
                
                # Save with compression
                encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_new.data_vars}
                ds_new.to_netcdf(self.archive_file, encoding=encoding)
                ds_new.close()
                
            else:
                # Many batches - hierarchical merge to avoid memory exhaustion
                logger.info(f"  Many batches ({len(new_batches)}) - using hierarchical merge...")
                
                # Merge in groups of 5
                group_size = 5
                temp_merged = []
                
                for i in range(0, len(new_batches), group_size):
                    group = new_batches[i:i+group_size]
                    group_num = i // group_size + 1
                    total_groups = (len(new_batches) - 1) // group_size + 1
                    
                    logger.info(f"    Merging group {group_num}/{total_groups} ({len(group)} files)...")
                    
                    # Merge this group with preprocessing
                    ds_group = xr.open_mfdataset(
                        [str(f) for f in group],
                        combine='nested',
                        concat_dim='location',
                        parallel=False,
                        engine='netcdf4',
                        join='outer',  # Allow different dimension sizes
                        preprocess=self._preprocess_for_merge  # Normalize before merging
                    )
                    
                    # Save temp merged file
                    temp_file = self.temp_dir / f'hierarchical_merge_{group_num:03d}.nc'
                    encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_group.data_vars}
                    ds_group.to_netcdf(temp_file, encoding=encoding)
                    ds_group.close()
                    
                    temp_merged.append(temp_file)
                    gc.collect()
                
                # Now merge the temp files
                logger.info(f"  Final merge of {len(temp_merged)} groups...")
                ds_final = xr.open_mfdataset(
                    [str(f) for f in temp_merged],
                    combine='nested',
                    concat_dim='location',
                    parallel=False,
                    engine='netcdf4',
                    join='outer',  # Allow different dimension sizes
                    preprocess=self._preprocess_for_merge  # Normalize datasets
                )
                
                encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_final.data_vars}
                ds_final.to_netcdf(self.archive_file, encoding=encoding)
                ds_final.close()
                
                # Clean up temp files
                for f in temp_merged:
                    f.unlink()
                
                gc.collect()
            
            size_mb = self.archive_file.stat().st_size / (1024**2)
            logger.info(f"  ✅ Created: {size_mb:.1f} MB")
            
        else:
            # Existing archive - merge (for operational use, we typically just replace)
            logger.info("  Existing archive found")
            logger.info("  For operational archive, replacing with new data (current state)")
            
            # For operational archive, we don't accumulate - we replace
            # Just create new archive from batches
            logger.info("  Creating updated archive from new batches...")
            
            if len(new_batches) <= 5:
                # Few batches - merge directly
                ds_new = xr.open_mfdataset(
                    [str(f) for f in new_batches],
                    combine='nested',
                    concat_dim='location',
                    parallel=False,
                    engine='netcdf4',
                    join='outer',
                    preprocess=self._preprocess_for_merge
                )
                
                # Save with compression
                temp_archive = self.archive_file.with_suffix('.nc.tmp')
                encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_new.data_vars}
                ds_new.to_netcdf(temp_archive, encoding=encoding)
                ds_new.close()
                
                # Atomic replace
                temp_archive.replace(self.archive_file)
                
            else:
                # Many batches - hierarchical merge
                logger.info(f"  Many batches ({len(new_batches)}) - using hierarchical merge...")
                
                group_size = 5
                temp_merged = []
                
                for i in range(0, len(new_batches), group_size):
                    group = new_batches[i:i+group_size]
                    group_num = i // group_size + 1
                    total_groups = (len(new_batches) - 1) // group_size + 1
                    
                    logger.info(f"    Merging group {group_num}/{total_groups} ({len(group)} files)...")
                    
                    ds_group = xr.open_mfdataset(
                        [str(f) for f in group],
                        combine='nested',
                        concat_dim='location',
                        parallel=False,
                        engine='netcdf4',
                        join='outer',
                        preprocess=self._preprocess_for_merge
                    )
                    
                    temp_file = self.temp_dir / f'hierarchical_merge_{group_num:03d}.nc'
                    encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_group.data_vars}
                    ds_group.to_netcdf(temp_file, encoding=encoding)
                    ds_group.close()
                    
                    temp_merged.append(temp_file)
                    gc.collect()
                
                # Final merge
                logger.info(f"  Final merge of {len(temp_merged)} groups...")
                ds_final = xr.open_mfdataset(
                    [str(f) for f in temp_merged],
                    combine='nested',
                    concat_dim='location',
                    parallel=False,
                    engine='netcdf4',
                    join='outer',
                    preprocess=self._preprocess_for_merge
                )
                
                temp_archive = self.archive_file.with_suffix('.nc.tmp')
                encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_final.data_vars}
                ds_final.to_netcdf(temp_archive, encoding=encoding)
                ds_final.close()
                
                # Clean up temp files
                for f in temp_merged:
                    f.unlink()
                
                # Atomic replace
                temp_archive.replace(self.archive_file)
                
                gc.collect()
            
            size_mb = self.archive_file.stat().st_size / (1024**2)
            logger.info(f"  ✅ Updated: {size_mb:.1f} MB")
    
    def _apply_time_window(self):
        """Apply rolling time window to archive."""
        if not self.archive_file.exists():
            return
        
        logger.info("")
        logger.info(f"Applying {self.window_days}-day rolling window...")
        
        ds = xr.open_dataset(self.archive_file)
        
        before_timesteps = len(ds.time)
        
        # Apply window
        cutoff = datetime.now() - timedelta(days=self.window_days)
        ds_windowed = ds.sel(time=slice(cutoff.isoformat(), None))
        
        after_timesteps = len(ds_windowed.time)
        removed = before_timesteps - after_timesteps
        
        if removed > 0:
            logger.info(f"  Removed {removed:,} old timesteps")
            
            # Save windowed version
            temp_file = self.archive_file.with_suffix('.nc.tmp')
            ds_windowed.to_netcdf(temp_file)
            
            ds.close()
            ds_windowed.close()
            
            temp_file.replace(self.archive_file)
        else:
            logger.info(f"  All timesteps within window")
            ds.close()
            ds_windowed.close()
        
        del ds, ds_windowed
        gc.collect()
    
    def _cleanup(self):
        """Clean up temp files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _update_tracking(self):
        """
        Update file modification times.
        
        IMPORTANT: Don't update tracking for files that failed to process!
        This ensures they'll be detected as "changed" on the next run.
        """
        all_files = list(self.data_dir.rglob('*.pro')) + list(self.data_dir.rglob('*.smet'))
        
        updated_count = 0
        skipped_count = 0
        
        for filepath in all_files:
            try:
                file_key = str(filepath.relative_to(self.data_dir))
                
                # Skip files that failed to process - they'll be retried next run
                if hasattr(self, 'failed_files') and file_key in self.failed_files:
                    skipped_count += 1
                    continue
                
                mtime = filepath.stat().st_mtime
                self.file_tracking[file_key] = mtime
                updated_count += 1
                
            except:
                continue
        
        with open(self.tracking_file, 'w') as f:
            json.dump(self.file_tracking, f, indent=2)
        
        if skipped_count > 0:
            logger.info(f"  Updated tracking for {updated_count:,} files")
            logger.info(f"  Skipped {skipped_count:,} failed files (will retry next run)")
        
        # Clear failed files set for next run
        if hasattr(self, 'failed_files'):
            delattr(self, 'failed_files')
    
    def _preprocess_for_merge(self, ds):
        """
        Preprocess datasets for merging.
        
        Handles both old and new batch file formats:
        - Old format (before xsnow fix): azimuth/inclination as 1D (slope,) coordinates
        - New format (after xsnow fix): azimuth/inclination as 2D (location, slope) data variables
        
        This defensive preprocessing ensures compatibility during the transition period.
        """
        # Convert old-format 1D slope coordinates to data variables
        slope_varying_coords = ['azimuth', 'aspect', 'inclination']
        
        for coord in slope_varying_coords:
            if coord in ds.coords:
                # Check if it's a problematic 1D slope coordinate (old format)
                if ds.coords[coord].dims == ('slope',):
                    # Old format - convert to data variable
                    # xarray will handle broadcasting during merge with join='outer'
                    ds = ds.reset_coords(coord)
        
        # Drop non-essential coordinates that might conflict
        essential = ['location', 'time', 'layer', 'slope', 'realization',
                     'lat', 'lon', 'latitude', 'longitude',
                     'x', 'y', 'z', 'altitude', 'elevation']
        
        coords_to_drop = [c for c in ds.coords if c not in essential]
        if coords_to_drop:
            ds = ds.drop_vars(coords_to_drop, errors='ignore')
        
        return ds
    
    def _show_status(self):
        """Show archive status."""
        if not self.archive_file.exists():
            logger.info("")
            logger.info("No operational archive yet")
            return
        
        ds = xr.open_dataset(self.archive_file)
        
        size_mb = self.archive_file.stat().st_size / (1024**2)
        
        logger.info("")
        logger.info("Operational archive:")
        logger.info(f"  File: {self.archive_file}")
        logger.info(f"  Size: {size_mb:.1f} MB")
        logger.info(f"  Locations: {len(ds.location):,}")
        logger.info(f"  Timesteps: {len(ds.time):,}")
        logger.info(f"  Date range: {ds.time.values[0]} to {ds.time.values[-1]}")
        
        ds.close()
        
        logger.info("")
        logger.info("Query with:")
        logger.info(f"  ds = xr.open_dataset('{self.archive_file}')")
        logger.info(f"  latest = ds.HS.isel(time=-1).compute()")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Operational SNOWPACK archive - current state approach",
        epilog="""
Perfect for 4 runs/day:
  6 AM:  Update operational.nc
  12 PM: Update operational.nc (replaces old)
  6 PM:  Update operational.nc (replaces old)
  12 AM: Update operational.nc (replaces old)

Result: ONE file, always current, no redundancy!

Examples:
  # Update operational archive (run 4x/day)
  python operational_archive.py /data/snowpack --window-days 14
  
  # First run might take 1-2 hours (all files)
  # Subsequent runs: 5-10 minutes (only changed files)
        """
    )
    
    parser.add_argument('data_dir')
    parser.add_argument('--archive', default='operational.nc')
    parser.add_argument('--window-days', type=int, default=14)
    parser.add_argument('--batch-size', type=int, default=2000)  # ✅ Reduced from 10,000
    parser.add_argument('--cpus', type=int, default=3)
    
    args = parser.parse_args()
    
    archive = OperationalArchive(
        data_dir=args.data_dir,
        archive_file=args.archive,
        window_days=args.window_days,
        batch_size=args.batch_size,
        n_cpus=args.cpus
    )
    
    archive.update()


if __name__ == '__main__':
    main()