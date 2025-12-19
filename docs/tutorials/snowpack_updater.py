#!/usr/bin/env python3
"""
SnowpackUpdater - Automatic Initial Load and Incremental Updates

Simple, intelligent class that:
- Auto-detects first run vs update
- First run: Keeps ALL historical data
- Updates: Applies rolling time window
- Handles both .pro and .smet files

Usage:
    updater = SnowpackUpdater(
        data_dir='/home/snowpack/runs/output.2024',
        archive_file='snowpack_operational.nc',
        window_days=90
    )
    updater.run()
"""

import xsnow
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('snowpack_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SnowpackUpdater:
    """
    Intelligent SNOWPACK archive updater.
    
    Auto-detects whether this is:
    - Initial run (no archive exists) → Keep all historical data
    - Update run (archive exists) → Apply rolling time window
    
    Example:
        # First run: Creates full historical archive
        updater = SnowpackUpdater('snowpack.nc', window_days=90)
        updater.run()  # Detects no archive, keeps ALL history
        
        # Later runs: Updates with 90-day window
        updater = SnowpackUpdater('snowpack.nc', window_days=90)
        updater.run()  # Detects archive exists, applies 90-day window
    """
    
    def __init__(
        self,
        data_dir: str,
        archive_file: str = 'snowpack_operational.nc',
        window_days: int = 14,
        mode: str = 'auto',
        lazy: bool = True,
        n_cpus: int = 15
    ):
        """
        Initialize SNOWPACK updater.
        
        Args:
            data_dir: Directory with .pro and .smet files
            archive_file: Output NetCDF archive path
            window_days: Days to keep in rolling window (for updates)
            mode: 'auto', 'initial', or 'update'
                  'auto' = auto-detect based on archive existence
                  'initial' = force initial load (ignore existing archive)
                  'update' = force update mode (require existing archive)
            lazy: Use lazy loading (recommended for large datasets)
            n_cpus: Number of CPUs to use for parallel loading
        """
        self.data_dir = Path(data_dir)
        self.archive_file = Path(archive_file)
        self.user_window_days = window_days
        self.lazy = lazy
        self.n_cpus = n_cpus
        
        # Determine mode
        if mode == 'auto':
            # Auto-detect based on archive existence
            if self.archive_file.exists():
                self.mode = 'update'
                self.window_days = window_days
            else:
                self.mode = 'initial'
                self.window_days = 0  # Keep all history!
        elif mode == 'initial':
            self.mode = 'initial'
            self.window_days = 0  # Keep all history
        elif mode == 'update':
            if not self.archive_file.exists():
                raise FileNotFoundError(
                    f"Update mode specified but archive not found: {self.archive_file}"
                )
            self.mode = 'update'
            self.window_days = window_days
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'auto', 'initial', or 'update'")
        
        # Stats
        self.start_time = None
        self.n_pro = 0
        self.n_smet = 0
        self.n_locations = 0
        self.n_timesteps = 0
    
    def run(self):
        """
        Run the update process.
        
        Returns:
            xsnowDataset with processed data
        """
        self.start_time = time.time()
        
        self._print_header()
        self._set_low_priority()
        self._count_and_estimate()
        
        # Load data
        ds = self._load_data()
        if ds is None:
            logger.error("❌ Failed to load data")
            return None
        
        # Apply time window (if update mode)
        ds = self._apply_time_window(ds)
        
        # Process data
        ds = self._process_data(ds)
        
        # Save archive
        self._save_archive(ds)
        
        # Summary
        self._print_summary()
        
        return ds
    
    def _print_header(self):
        """Print startup header with mode information."""
        logger.info("="*70)
        logger.info("SNOWPACK ARCHIVE UPDATER")
        logger.info("="*70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Archive file: {self.archive_file}")
        logger.info(f"")
        logger.info(f"🔍 Mode: {self.mode.upper()}")
        
        if self.mode == 'initial':
            logger.info(f"   First run detected - will create full historical archive")
            logger.info(f"   Keeping ALL historical data (no time window)")
        else:
            logger.info(f"   Archive exists - updating with {self.window_days}-day window")
            logger.info(f"   Last modified: {datetime.fromtimestamp(self.archive_file.stat().st_mtime)}")
        
        logger.info("="*70)
    
    def _set_low_priority(self):
        """Set process to low CPU priority to not interfere with SNOWPACK."""
        try:
            os.nice(10)
            logger.info("⚙️  Set to low CPU priority (nice +10)")
        except:
            pass
    
    def _count_and_estimate(self):
        """Count files and estimate resource requirements."""
        logger.info("\n📊 Scanning for files...")
        
        # Count files
        pro_files = list(self.data_dir.rglob('*.pro'))
        smet_files = list(self.data_dir.rglob('*.smet'))
        self.n_pro = len(pro_files)
        self.n_smet = len(smet_files)
        n_total = self.n_pro + self.n_smet
        
        logger.info(f"   .pro files:  {self.n_pro:,}")
        logger.info(f"   .smet files: {self.n_smet:,}")
        logger.info(f"   Total:       {n_total:,}")
        
        # Memory estimate
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if self.lazy:
            estimated_gb = 2  # Lazy uses minimal memory
        else:
            # .pro: ~0.6 MB, .smet: ~0.1 MB in memory
            estimated_gb = (self.n_pro * 0.6 + self.n_smet * 0.1) / 1024
        
        logger.info(f"\n💾 Memory estimate:")
        logger.info(f"   Available: {available_gb:.1f} GB")
        logger.info(f"   Required:  {estimated_gb:.1f} GB")
        
        if estimated_gb > available_gb * 0.7:
            logger.warning(f"   ⚠️  Memory usage may be high!")
            if not self.lazy:
                logger.warning(f"   Consider using lazy=True")
    
    def _load_data(self):
        """Load all .pro and .smet files."""
        logger.info(f"\n⚙️  Loading data...")
        logger.info(f"   Mode: {'Lazy' if self.lazy else 'Eager'}")
        logger.info(f"   CPUs: {self.n_cpus}")
        logger.info(f"   This will take 30-60 minutes...")
        
        load_start = time.time()
        
        try:
            ds = xsnow.read(
                self.data_dir,
                recursive=True,
                lazy=self.lazy,
                n_cpus_use=self.n_cpus,
                parallel_lazy_creation=True
            )
            
            if ds is None:
                return None
            
            load_time = time.time() - load_start
            self.n_locations = len(ds.data.location)
            self.n_timesteps = len(ds.data.time)
            
            logger.info(f"   ✅ Loaded in {load_time/60:.1f} minutes")
            logger.info(f"   Locations: {self.n_locations:,}")
            logger.info(f"   Time points: {self.n_timesteps:,}")
            logger.info(f"   Time range: {ds.data.time.values[0]} to {ds.data.time.values[-1]}")
            
            return ds
            
        except Exception as e:
            logger.error(f"   ❌ Failed to load data: {e}")
            return None
    
    def _apply_time_window(self, ds):
        """Apply time window if in update mode."""
        if self.window_days == 0:
            logger.info(f"\n📅 Keeping all historical data (initial mode)")
            return ds
        
        logger.info(f"\n📅 Applying {self.window_days}-day time window...")
        
        cutoff = datetime.now() - timedelta(days=self.window_days)
        cutoff_str = cutoff.isoformat()
        
        logger.info(f"   Cutoff date: {cutoff_str[:10]}")
        
        ds_windowed = ds.sel(time=slice(cutoff_str, None))
        n_times_after = len(ds_windowed.data.time)
        n_times_before = len(ds.data.time)
        n_removed = n_times_before - n_times_after
        
        logger.info(f"   Time points before: {n_times_before:,}")
        logger.info(f"   Time points after:  {n_times_after:,}")
        logger.info(f"   Removed: {n_removed:,} old timesteps")
        
        return ds_windowed
    
    def _process_data(self, ds):
        """Process north aspects and calculate max HS."""
        logger.info(f"\n🧭 Processing data...")
        
        # Filter for north aspects
        if 'aspect' in ds.data:
            logger.info("   Filtering north aspects (315° - 45°)...")
            north_mask = ((ds.data.aspect >= 315) | (ds.data.aspect <= 45))
            ds_north = ds.where(north_mask, drop=True)
            
            n_north = len(ds_north.data.location)
            n_total = len(ds.data.location)
            pct = (n_north / n_total * 100) if n_total > 0 else 0
            
            logger.info(f"   North-facing: {n_north:,} / {n_total:,} ({pct:.1f}%)")
            ds = ds_north
        else:
            logger.warning("   No 'aspect' variable - using all locations")
        
        # Calculate max HS
        if 'HS' in ds.data:
            logger.info("   Calculating max HS...")
            max_hs = ds.data.HS.max(dim='time')
            
            # Try to get stats if already computed
            try:
                if not hasattr(max_hs.data, 'compute'):
                    hs_min = float(max_hs.min().values)
                    hs_max = float(max_hs.max().values)
                    hs_mean = float(max_hs.mean().values)
                    logger.info(f"   Max HS range: {hs_min:.2f} to {hs_max:.2f} m (mean: {hs_mean:.2f} m)")
            except:
                pass
            
            ds.data['max_HS'] = max_hs
        else:
            logger.warning("   No 'HS' variable found")
        
        return ds
    
    def _save_archive(self, ds):
        """Save processed data to archive."""
        logger.info(f"\n💾 Saving archive to {self.archive_file}...")
        
        save_start = time.time()
        
        try:
            # Ensure parent directory exists
            self.archive_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save
            ds.to_netcdf(self.archive_file)
            
            save_time = time.time() - save_start
            file_size_mb = self.archive_file.stat().st_size / (1024**2)
            
            logger.info(f"   ✅ Saved in {save_time/60:.1f} minutes")
            logger.info(f"   File size: {file_size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to save archive: {e}")
            raise
    
    def _print_summary(self):
        """Print summary statistics."""
        total_time = time.time() - self.start_time
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ {self.mode.upper()} COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"")
        logger.info(f"Input:")
        logger.info(f"  .pro files:  {self.n_pro:,}")
        logger.info(f"  .smet files: {self.n_smet:,}")
        logger.info(f"")
        logger.info(f"Output:")
        logger.info(f"  Archive: {self.archive_file}")
        logger.info(f"  Size: {self.archive_file.stat().st_size / (1024**2):.1f} MB")
        logger.info(f"  Locations: {self.n_locations:,}")
        logger.info(f"  Time points: {self.n_timesteps:,}")
        
        if self.mode == 'initial':
            logger.info(f"")
            logger.info(f"📝 Next run will use UPDATE mode with {self.user_window_days}-day window")
        
        logger.info("="*70)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SNOWPACK Archive Updater with automatic mode detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect mode (recommended)
  %(prog)s /data/snowpack/pro
  
  # Force initial load
  %(prog)s /data/snowpack/pro --mode initial
  
  # Force update mode
  %(prog)s /data/snowpack/pro --mode update
  
  # Custom archive location and window
  %(prog)s /data/snowpack/pro --archive my_archive.nc --window-days 30
        """
    )
    
    parser.add_argument(
        'data_dir',
        help='Directory with .pro and .smet files'
    )
    parser.add_argument(
        '--archive',
        default='snowpack_operational.nc',
        help='Archive file path (default: snowpack_operational.nc)'
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=14,
        help='Time window in days for updates (default: 90)'
    )
    parser.add_argument(
        '--mode',
        choices=['auto', 'initial', 'update'],
        default='auto',
        help='Mode: auto-detect, force initial, or force update (default: auto)'
    )
    parser.add_argument(
        '--lazy',
        action='store_true',
        default=True,
        help='Use lazy loading (default: True)'
    )
    parser.add_argument(
        '--eager',
        action='store_true',
        help='Use eager loading instead of lazy'
    )
    parser.add_argument(
        '--cpus',
        type=int,
        default=15,
        help='Number of CPUs to use (default: 15)'
    )
    
    args = parser.parse_args()
    
    # Handle lazy vs eager
    use_lazy = not args.eager
    
    # Create and run updater
    updater = SnowpackUpdater(
        data_dir=args.data_dir,
        archive_file=args.archive,
        window_days=args.window_days,
        mode=args.mode,
        lazy=use_lazy,
        n_cpus=args.cpus
    )
    
    updater.run()


if __name__ == '__main__':
    main()
