#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNOWPACK Archive Consolidator - CLEAN & THOROUGH

Philosophy: Take your time, do it right!
- Runs weekly/monthly (not daily)
- Creates clean single-file archive
- Can use large batches (fewer to merge)
- Takes 3-4 hours - that's OK!

This is for RESEARCH & LONG-TERM STORAGE - cleanliness matters!
"""

import xsnow
import xarray as xr
import time
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


class ArchiveConsolidator:
    """
    Clean archive consolidation.
    
    Takes time (3-4 hours) but creates perfect clean archive.
    Run weekly/monthly when you have time and memory.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_file: str,
        window_days: int = 365,
        batch_size: int = 10000,  # Larger batches = fewer to merge!
        n_cpus: int = 3,
        temp_dir: str = 'temp_archive'
    ):
        self.data_dir = Path(data_dir)
        self.output_file = Path(output_file)
        self.window_days = window_days
        self.batch_size = batch_size
        self.n_cpus = n_cpus
        self.temp_dir = Path(temp_dir)
    
    def create_archive(self, from_source=True, operational_dir=None):
        """
        Create clean consolidated archive.
        
        Args:
            from_source: If True, read from .pro/.smet files
                        If False, use operational batches
            operational_dir: Path to operational archive (if from_source=False)
        """
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("ARCHIVE CONSOLIDATION - CLEAN & THOROUGH")
        logger.info("="*70)
        logger.info(f"Output: {self.output_file}")
        logger.info(f"Window: {self.window_days} days")
        logger.info(f"Batch size: {self.batch_size:,} files (larger = fewer batches)")
        logger.info("")
        
        if from_source:
            logger.info("Strategy: Reading from source .pro/.smet files")
            logger.info("  (Cleanest, but takes longer)")
        else:
            logger.info(f"Strategy: Consolidating from {operational_dir}")
            logger.info("  (Faster, uses existing operational batches)")
        
        logger.info("")
        logger.info("⏰ This will take 3-4 hours - go get coffee! ☕")
        logger.info("")
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        
        # Process
        if from_source:
            self._process_from_source()
        else:
            self._process_from_operational(operational_dir)
        
        # Combine into single file
        self._combine_batches()
        
        # Clean up
        shutil.rmtree(self.temp_dir)
        
        total_time = time.time() - start_time
        
        logger.info("")
        logger.info("="*70)
        logger.info("✅ ARCHIVE COMPLETE")
        logger.info("="*70)
        logger.info(f"Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        logger.info(f"Archive: {self.output_file}")
        
        size_gb = self.output_file.stat().st_size / (1024**3)
        logger.info(f"Size: {size_gb:.2f} GB")
        logger.info("")
        logger.info("This is a CLEAN, SINGLE-FILE archive ✅")
        logger.info("Perfect for research and long-term storage!")
    
    def _process_from_source(self):
        """Process from source .pro/.smet files."""
        logger.info("Scanning source files...")
        
        all_files = list(self.data_dir.rglob('*.pro')) + list(self.data_dir.rglob('*.smet'))
        n_total = len(all_files)
        n_batches = (n_total + self.batch_size - 1) // self.batch_size
        
        logger.info(f"  Found: {n_total:,} files")
        logger.info(f"  Will create: {n_batches} batches ({self.batch_size:,} files each)")
        logger.info(f"  Fewer batches = easier final merge ✅")
        logger.info("")
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_total)
            batch_files = all_files[start_idx:end_idx]
            
            logger.info(f"Batch {i+1}/{n_batches}: {len(batch_files):,} files")
            
            try:
                ds = xsnow.read(
                    batch_files,
                    lazy=True,
                    n_cpus_use=self.n_cpus,
                    parallel_lazy_creation=False
                )
                
                if ds is None:
                    logger.warning("  ⚠️  No data")
                    continue
                
                # Apply window
                if self.window_days > 0:
                    cutoff = datetime.now() - timedelta(days=self.window_days)
                    ds = ds.sel(time=slice(cutoff.isoformat(), None))
                
                # Save
                batch_file = self.temp_dir / f'batch_{i:04d}.nc'
                ds.to_netcdf(batch_file)
                
                size_mb = batch_file.stat().st_size / (1024**2)
                logger.info(f"  ✅ {size_mb:.1f} MB")
                
                del ds
                gc.collect()
                
            except Exception as e:
                logger.error(f"  ❌ Failed: {e}")
    
    def _process_from_operational(self, operational_dir):
        """Consolidate from operational batches (faster)."""
        logger.info(f"Loading operational batches from {operational_dir}...")
        
        operational_batches = list(Path(operational_dir).glob('batch_*.nc'))
        n_batches = len(operational_batches)
        
        logger.info(f"  Found: {n_batches} operational batches")
        logger.info("")
        logger.info("Opening all batches with Dask...")
        
        # Use open_mfdataset to load all operational batches
        try:
            ds_all = xr.open_mfdataset(
                [str(b) for b in operational_batches],
                combine='nested',
                concat_dim='location',
                parallel=True
            )
            
            logger.info("  ✅ All batches loaded")
            
            # Apply window
            if self.window_days > 0:
                cutoff = datetime.now() - timedelta(days=self.window_days)
                ds_all = ds_all.sel(time=slice(cutoff.isoformat(), None))
            
            # Save directly as final archive
            logger.info("")
            logger.info("Saving consolidated archive...")
            logger.info("  (This step processes everything with Dask)")
            
            ds_all.to_netcdf(self.output_file, compute=True)
            
            logger.info("  ✅ Archive saved")
            
            ds_all.close()
            del ds_all
            gc.collect()
            
            # Mark as already combined
            self._already_combined = True
            
        except Exception as e:
            logger.error(f"Failed to consolidate operational batches: {e}")
            logger.error("Try from_source=True instead")
            raise
    
    def _combine_batches(self):
        """Combine temp batches into single archive."""
        
        # Check if already combined (from operational)
        if hasattr(self, '_already_combined'):
            logger.info("Already combined from operational batches ✅")
            return
        
        batch_files = sorted(self.temp_dir.glob('batch_*.nc'))
        n_batches = len(batch_files)
        
        if n_batches == 0:
            logger.error("No batches to combine!")
            return
        
        logger.info("")
        logger.info("="*70)
        logger.info("FINAL COMBINE")
        logger.info("="*70)
        logger.info(f"Combining {n_batches} batches into single file...")
        logger.info("")
        
        if n_batches <= 20:
            # Few enough to combine directly
            logger.info("Using open_mfdataset (Dask-powered)...")
            
            ds_combined = xr.open_mfdataset(
                [str(f) for f in batch_files],
                combine='nested',
                concat_dim='location',
                parallel=True
            )
            
            logger.info("  Saving final archive...")
            ds_combined.to_netcdf(self.output_file, compute=True)
            
            ds_combined.close()
            
        else:
            # Many batches - combine in chunks
            logger.info(f"Many batches ({n_batches}) - combining in chunks...")
            
            chunk_size = 20
            chunks = [batch_files[i:i+chunk_size] for i in range(0, n_batches, chunk_size)]
            
            intermediate_files = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"  Chunk {i+1}/{len(chunks)}: {len(chunk)} batches")
                
                ds_chunk = xr.open_mfdataset(
                    [str(f) for f in chunk],
                    combine='nested',
                    concat_dim='location',
                    parallel=True
                )
                
                intermediate_file = self.temp_dir / f'intermediate_{i:04d}.nc'
                ds_chunk.to_netcdf(intermediate_file, compute=True)
                intermediate_files.append(intermediate_file)
                
                ds_chunk.close()
                del ds_chunk
                gc.collect()
            
            # Final combine
            logger.info(f"  Final combine of {len(intermediate_files)} intermediate files...")
            
            ds_final = xr.open_mfdataset(
                [str(f) for f in intermediate_files],
                combine='nested',
                concat_dim='location',
                parallel=True
            )
            
            ds_final.to_netcdf(self.output_file, compute=True)
            ds_final.close()
        
        logger.info("  ✅ Combined into single file")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create clean consolidated SNOWPACK archive",
        epilog="""
Examples:
  # From source files (cleanest, 3-4 hours)
  python snowpack_archiver.py /data/snowpack --output archive_2024.nc
  
  # From operational batches (faster, 1-2 hours)
  python snowpack_archiver.py --from-operational operational_archive \
      --output archive_2024.nc
  
  # Full year
  python snowpack_archiver.py /data/snowpack \
      --window-days 365 \
      --output archive_2024_full.nc
  
  # Winter only
  python snowpack_archiver.py /data/snowpack \
      --window-days 120 \
      --output archive_2024_winter.nc
        """
    )
    
    parser.add_argument('data_dir', nargs='?')
    parser.add_argument('--output', required=True, help='Output archive file')
    parser.add_argument('--window-days', type=int, default=365)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--cpus', type=int, default=3)
    parser.add_argument('--from-operational', help='Consolidate from operational archive')
    
    args = parser.parse_args()
    
    consolidator = ArchiveConsolidator(
        data_dir=args.data_dir if args.data_dir else '.',
        output_file=args.output,
        window_days=args.window_days,
        batch_size=args.batch_size,
        n_cpus=args.cpus
    )
    
    if args.from_operational:
        # Consolidate from operational batches (faster)
        consolidator.create_archive(
            from_source=False,
            operational_dir=args.from_operational
        )
    elif args.data_dir:
        # From source files (cleaner)
        consolidator.create_archive(from_source=True)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
