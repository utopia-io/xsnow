#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNOWPACK Operational Archive - Current State Approach - ZARR VERSION

Key insight: Operational archive = CURRENT STATE, not history!

Instead of creating new batches every run (4x/day = tons of redundancy),
maintain ONE operational archive that gets updated with latest data.

Perfect for 4 runs/day scenario:
- 6 AM: Update operational.zarr with latest
- 12 PM: Update operational.zarr with latest (replaces old timestamps)
- 6 PM: Update operational.zarr with latest
- 12 AM: Update operational.zarr with latest

Result: ONE file, always current, no redundancy!

Zarr advantages:
- Better stability (pure Python, no HDF5 crashes)
- Faster compression (blosc/zstd vs zlib)
- Lower memory usage (streaming writes)
- Cloud-friendly format
"""

import xsnow
import xarray as xr
import numpy as np
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
    Maintains current operational state using Zarr format.
    
    Updates existing archive rather than creating new batches.
    Perfect for frequent runs (4x/day).
    """
    
    def __init__(
        self,
        data_dir: str,
        archive_file: str = 'operational.zarr',
        tracking_file: str = 'file_tracking.json',
        window_days: int = 14,
        batch_size: int = 20000,
        n_cpus: int = 3,
        modification_threshold: int = 60
    ):
        """
        Operational archive for SNOWPACK using Zarr format.
        
        Args:
            data_dir: Directory containing SNOWPACK output files
            archive_file: Output Zarr archive path
            tracking_file: JSON file tracking processed files
            window_days: Keep this many days of data (rolling window)
            batch_size: Files per batch for processing
            n_cpus: CPUs for parallel processing
            modification_threshold: Skip files modified within this many seconds
        """
        self.data_dir = Path(data_dir)
        self.archive_file = Path(archive_file)
        self.tracking_file = Path(tracking_file)
        self.window_days = window_days
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.modification_threshold = modification_threshold
        
        # Ensure .zarr extension
        if not str(self.archive_file).endswith('.zarr'):
            self.archive_file = Path(str(self.archive_file).replace('.nc', '.zarr'))
            logger.info(f"Using Zarr format: {self.archive_file}")
        
        self.file_tracking = {}
        self.failed_files = set()
        self.temp_dir = Path('temp_operational')
    
    def _get_zarr_encoding(self, ds):
        """
        Get Zarr encoding with fill values for integer types.
        
        Uses SNOWPACK standard (-999) when possible, dtype-specific otherwise.
        """
        FILL_VALUES = {
            'int8': -128,      # Closest to -999 that fits
            'int16': -999,     # SNOWPACK standard
            'int32': -999,     # SNOWPACK standard
            'int64': -999,     # SNOWPACK standard
            'uint8': 255,
            'uint16': 65535,
            'uint32': 4294967295,
            'uint64': 18446744073709551615,
        }
        
        encoding = {}
        for var in ds.data_vars:
            dtype_str = str(ds[var].dtype)
            if 'int' in dtype_str:
                fill_value = FILL_VALUES.get(dtype_str, -999)
                encoding[var] = {'_FillValue': fill_value}
        
        return encoding
    
    def _save_to_zarr(self, ds, output_path, mode='w'):
        """
        Save dataset to Zarr with proper encoding.
        
        Handles fill values for integer types.
        Zarr v3 applies default compression automatically.
        """
        encoding = self._get_zarr_encoding(ds)
        
        if encoding:
            logger.debug(f"Saving with fill values for {len(encoding)} integer variables")
            ds.to_zarr(output_path, mode=mode, encoding=encoding)
        else:
            ds.to_zarr(output_path, mode=mode)
    
    def _get_optimal_chunks(self, ds):
        """
        Get optimal chunk sizes based on dataset dimensions.
        
        Only chunks dimensions that exist (handles both profile and time-series data).
        """
        desired_chunks = {
            'location': 200,
            'time': 100,
            'layer': -1,
            'slope': -1,
            'realization': -1
        }
        
        chunks = {}
        for dim, chunk_size in desired_chunks.items():
            if dim in ds.dims:
                chunks[dim] = chunk_size
        
        logger.debug(f"Chunking with: {chunks}")
        return chunks
    
    def _preprocess_for_merge(self, ds):
        """
        Preprocess datasets for merging.
        
        1. Converts slope coordinates to data variables
        2. Drops xsnow-derived and non-essential coordinates
        """
        # Always convert slope coordinates to data variables
        slope_coords = ['azimuth', 'inclination', 'aspect']
        
        for coord in slope_coords:
            if coord in ds.coords:
                ds = ds.reset_coords(coord)
                
                # Broadcast 1D to 2D if needed
                coord_var = ds[coord]
                if coord_var.dims == ('slope',) and 'location' in ds.dims and 'slope' in ds.dims:
                    template = None
                    for var_name in ds.data_vars:
                        if var_name == coord:
                            continue
                        var = ds[var_name]
                        if 'location' in var.dims and 'slope' in var.dims:
                            template = var
                            for dim in var.dims:
                                if dim not in ['location', 'slope']:
                                    template = template.isel({dim: 0})
                            break
                    
                    if template is not None:
                        ds[coord] = ds[coord].broadcast_like(template)
        
        # Drop derived and non-essential coordinates
        essential_coords = [
            'location', 'time', 'layer', 'slope', 'realization',
            'latitude', 'longitude', 'altitude', 'elevation',
            'lat', 'lon'
        ]
        
        coords_to_drop = [c for c in ds.coords if c not in essential_coords]
        if coords_to_drop:
            logger.debug(f"Dropping coordinates: {coords_to_drop}")
            ds = ds.drop_vars(coords_to_drop, errors='ignore')
        
        return ds
    
    def update(self):
        """Update operational archive with latest SNOWPACK data."""
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("OPERATIONAL ARCHIVE UPDATE - ZARR VERSION")
        logger.info("="*70)
        logger.info(f"Archive: {self.archive_file}")
        logger.info(f"Window: {self.window_days} days (rolling)")
        logger.info(f"Format: Zarr (better stability, compression)")
        logger.info("")
        
        # Load tracking
        self._load_tracking()
        
        # Find changed files
        changed_files = self._find_changed_files()
        
        if not changed_files:
            logger.info("✅ No changes - archive is current")
            
            if self.archive_file.exists():
                self._apply_time_window()
            
            self._show_status()
            return
        
        logger.info(f"Found {len(changed_files):,} changed files")
        
        total_files = len(list(self.data_dir.rglob('*.pro'))) + len(list(self.data_dir.rglob('*.smet')))
        pct_changed = (len(changed_files) / total_files * 100) if total_files > 0 else 0
        
        if pct_changed > 90:
            logger.info(f"  ({pct_changed:.0f}% of all files - SNOWPACK updated everything)")
        
        logger.info("")
        
        # Process changed data
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
        """Find files that changed since last run."""
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
                    continue
                
                # Skip very recent files (likely still being written)
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
        """Create temporary Zarr batches with new data."""
        self.temp_dir.mkdir(exist_ok=True)
        
        # Split into batches
        batches = [changed_files[i:i + self.batch_size] 
                  for i in range(0, len(changed_files), self.batch_size)]
        
        logger.info(f"Processing {len(batches)} batch(es)...")
        
        batch_files = []
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"Batch {i}/{len(batches)}: {len(batch)} files")
            
            try:
                # Read with xsnow
                ds = xsnow.read(
                    [str(f) for f in batch],
                    lazy=False,
                    n_workers=self.n_cpus,
                    logger=logger
                )
                
                if ds is None or ds.data is None:
                    logger.warning(f"  Batch {i} returned no data")
                    continue
                
                ds_raw = ds.data
                logger.info(f"  Loaded: {len(ds_raw.location):,} locations")
                
                # Preprocess
                ds_raw = self._preprocess_for_merge(ds_raw)
                
                # Rechunk
                chunks = self._get_optimal_chunks(ds_raw)
                ds_rechunked = ds_raw.chunk(chunks)
                
                # Save to temp Zarr
                batch_file = self.temp_dir / f'batch_{i:04d}.zarr'
                logger.info(f"  Saving to {batch_file.name}")
                self._save_to_zarr(ds_rechunked, batch_file)
                batch_files.append(batch_file)
                
                # Clean up
                del ds, ds_raw, ds_rechunked
                gc.collect()
                
                logger.info(f"  ✅ Batch {i} complete")
                
            except Exception as e:
                logger.error(f"  Failed batch {i}: {e}")
                continue
        
        return batch_files
    
    def _merge_with_archive(self, new_batches):
        """Merge new data with existing archive."""
        logger.info("")
        logger.info("Merging with archive...")
        
        # Load new data
        logger.info("Loading new data...")
        new_datasets = []
        for batch_file in new_batches:
            ds = xr.open_zarr(batch_file)
            ds = self._preprocess_for_merge(ds)
            new_datasets.append(ds)
        
        if len(new_datasets) == 1:
            new_data = new_datasets[0]
        else:
            new_data = xr.concat(new_datasets, 
                                 dim='location', 
                                 join='outer', 
                                 coords='minimal',
                                 data_vars='all')
        
        logger.info(f"  New data: {len(new_data.location):,} locations")
        
        # Load existing if present
        if self.archive_file.exists():
            logger.info("Loading existing archive...")
            old_data = xr.open_zarr(self.archive_file)
            old_data = self._preprocess_for_merge(old_data)
            logger.info(f"  Existing: {len(old_data.location):,} locations")
            
            # Combine
            logger.info("Combining datasets...")
            combined = xr.concat([old_data, new_data], 
                                 dim='time', 
                                 join='outer',
                                 coords='minimal', 
                                 data_vars='all')
            
            # Remove duplicates
            _, unique_indices = np.unique(combined.time.values, return_index=True)
            combined = combined.isel(time=sorted(unique_indices))
            
            old_data.close()
        else:
            logger.info("Creating new archive (no existing data)")
            combined = new_data
        
        logger.info(f"Combined: {len(combined.location):,} locations, {len(combined.time):,} times")
        
        # Rechunk and save
        logger.info("Saving merged archive...")
        chunks = self._get_optimal_chunks(combined)
        combined_rechunked = combined.chunk(chunks)
        
        # Remove old archive
        if self.archive_file.exists():
            shutil.rmtree(self.archive_file)
        
        # Save new archive
        self._save_to_zarr(combined_rechunked, self.archive_file)
        
        logger.info("  ✅ Archive updated")
        
        # Clean up
        combined.close()
        for ds in new_datasets:
            ds.close()
        del new_data, combined, combined_rechunked
        gc.collect()
    
    def _apply_time_window(self):
        """Remove data older than window_days."""
        if not self.archive_file.exists():
            return
        
        logger.info("")
        logger.info(f"Applying {self.window_days}-day time window...")
        
        ds = xr.open_zarr(self.archive_file)
        
        cutoff = datetime.now() - timedelta(days=self.window_days)
        cutoff_timestamp = np.datetime64(cutoff)
        
        # Filter times
        mask = ds.time.values >= cutoff_timestamp
        ds_windowed = ds.isel(time=mask)
        
        removed = len(ds.time) - len(ds_windowed.time)
        
        if removed > 0:
            logger.info(f"  Removing {removed} old timestep(s)")
            logger.info(f"  Keeping: {len(ds_windowed.time)} timesteps")
            
            # Rechunk and save
            chunks = self._get_optimal_chunks(ds_windowed)
            ds_rechunked = ds_windowed.chunk(chunks)
            
            # Remove old
            ds.close()
            ds_windowed.close()
            shutil.rmtree(self.archive_file)
            
            # Save windowed
            self._save_to_zarr(ds_rechunked, self.archive_file)
            
            del ds_rechunked
            gc.collect()
        else:
            ds.close()
            ds_windowed.close()
            logger.info("  All data within window")
    
    def _cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            logger.info("")
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(self.temp_dir)
    
    def _update_tracking(self):
        """Update file tracking with processed files."""
        current_time = time.time()
        
        all_files = list(self.data_dir.rglob('*.pro')) + list(self.data_dir.rglob('*.smet'))
        
        for filepath in all_files:
            try:
                file_key = str(filepath.relative_to(self.data_dir))
                mtime = filepath.stat().st_mtime
                
                if current_time - mtime >= self.modification_threshold:
                    self.file_tracking[file_key] = mtime
                    
            except Exception:
                continue
        
        with open(self.tracking_file, 'w') as f:
            json.dump(self.file_tracking, f, indent=2)
        
        logger.info(f"Updated tracking: {len(self.file_tracking):,} files")
    
    def _show_status(self):
        """Show archive status."""
        if not self.archive_file.exists():
            logger.info("")
            logger.info("Archive does not exist yet")
            return
        
        logger.info("")
        logger.info("Archive Status:")
        
        # Get size
        import subprocess
        result = subprocess.run(['du', '-sh', str(self.archive_file)],
                               capture_output=True, text=True)
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            logger.info(f"  Size: {size_str}")
        
        # Get dimensions
        ds = xr.open_zarr(self.archive_file)
        logger.info(f"  Locations: {len(ds.location):,}")
        logger.info(f"  Timesteps: {len(ds.time):,}")
        logger.info(f"  Variables: {len(ds.data_vars)}")
        
        # Date range
        times = ds.time.values
        logger.info(f"  Date range: {times[0]} to {times[-1]}")
        
        ds.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update operational SNOWPACK archive (Zarr format)'
    )
    
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing SNOWPACK output files'
    )
    
    parser.add_argument(
        '--archive',
        type=str,
        default='operational.zarr',
        help='Output Zarr archive path (default: operational.zarr)'
    )
    
    parser.add_argument(
        '--window-days',
        type=int,
        default=14,
        help='Keep this many days of data (default: 14)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5000,
        help='Files per batch (default: 5000)'
    )
    
    parser.add_argument(
        '--cpus',
        type=int,
        default=3,
        help='Number of CPUs for processing (default: 3)'
    )
    
    parser.add_argument(
        '--modification-threshold',
        type=int,
        default=60,
        help='Skip files modified within N seconds (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Create and run updater
    updater = OperationalArchive(
        data_dir=args.data_dir,
        archive_file=args.archive,
        window_days=args.window_days,
        batch_size=args.batch_size,
        n_cpus=args.cpus,
        modification_threshold=args.modification_threshold
    )
    
    updater.update()


if __name__ == '__main__':
    main()