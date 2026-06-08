#!/usr/bin/env python3
"""
SNOWPACK Archive Consolidation Script - Zarr Version with Auto-Detection

This version uses Zarr format and supports both batch merge and direct .pro processing.

Features:
- ✅ Auto-detects input type (batches vs .pro files)
- ✅ Zarr format (better stability than NetCDF)
- ✅ Direct .pro/.smet processing (no intermediate batches needed)
- ✅ Batch merge mode (lowest memory, fastest)
- ✅ Automatic slope coordinate handling (1D → 2D)
- ✅ Pure Python (no HDF5 crashes)

Zarr advantages:
- Better stability (pure Python, no C library crashes)
- Faster compression (blosc/zstd vs zlib)
- Lower memory usage (streaming writes)
- Cloud-friendly format
- Better per-location compression than NetCDF

Usage:
    # Auto-detect input type (recommended)
    python3 snowpack_archiver_zarr.py \\
        /data/snowpack \\
        --output archive_2024.zarr
    
    # Force batch mode
    python3 snowpack_archiver_zarr.py \\
        --from-batches temp_archive \\
        --output archive_2024.zarr
    
    # Direct .pro processing with custom batch size
    python3 snowpack_archiver_zarr.py \\
        /data/snowpack \\
        --output archive_2024.zarr \\
        --batch-size 2000
"""

import xarray as xr
import logging
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import gc
import shutil
import numcodecs

# Import shared utilities
from zarr_utils import (
    get_zarr_encoding,
    get_optimal_chunks,
    preprocess_for_merge,
    save_to_zarr,
    find_snowpack_files,
    filter_files_by_mtime,
    merge_zarr_batches,
    remove_duplicate_times,
    create_file_batches
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SNOWPACKArchiverZarr:
    """Archive SNOWPACK data using Zarr format."""
    
    def __init__(
        self,
        output_file,
        window_days=365,
        batch_size=2000,
        compression_level=5,
        create_transfer_copy=False,
        resume=False,
        parallel_merge=False
    ):
        self.output_file = Path(output_file)
        self.window_days = window_days
        self.batch_size = batch_size
        self.compression_level = compression_level
        self.create_transfer_copy = create_transfer_copy
        self.resume = resume
        self.parallel_merge = parallel_merge
        
        # Ensure .zarr extension
        if not str(self.output_file).endswith('.zarr'):
            self.output_file = Path(str(self.output_file).replace('.nc', '.zarr'))
            logger.info(f"Output will use Zarr format: {self.output_file}")
    
    def _process_pro_files(self, input_dir: Path):
        """
        Process .pro and .smet files directly to Zarr archive.
        
        This provides a single-step workflow without intermediate batch files.
        Uses xsnow.read() to handle file parsing and lazy loading.
        
        Args:
            input_dir: Directory containing .pro and/or .smet files
        """
        logger.info("=" * 70)
        logger.info("DIRECT .PRO/.SMET PROCESSING")
        logger.info("=" * 70)
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Batch size: {self.batch_size:,} files per iteration")
        logger.info(f"Strategy: Process in batches, write directly to Zarr")
        logger.info("")
        
        # Import xsnow
        try:
            import xsnow
        except ImportError:
            logger.error("xsnow package not found. Please install: pip install -e /path/to/xsnow")
            raise
        
        # Find all .pro and .smet files
        all_files = find_snowpack_files(input_dir, pattern='both')
        
        if not all_files:
            logger.error(f"No .pro or .smet files found in {input_dir}")
            raise ValueError("No input files found")
        
        # Separate by type for logging
        pro_files = [f for f in all_files if f.suffix == '.pro']
        smet_files = [f for f in all_files if f.suffix == '.smet']
        
        logger.info(f"Found {len(pro_files)} .pro files, {len(smet_files)} .smet files")
        logger.info(f"Total: {len(all_files):,} files to process")
        logger.info("")
        
        # Determine if we should process in batches or all at once
        if len(all_files) <= self.batch_size:
            # Small dataset - process all at once
            logger.info("Dataset is small enough to process in one batch")
            logger.info("Reading all files with xsnow.read()...")
            
            ds = xsnow.read(
                input_dir,
                recursive=True,
                lazy=False,  # Eager loading for small datasets
                logger=logger
            )
            
            if ds is None or ds.data is None:
                logger.error("Failed to read data")
                raise ValueError("xsnow.read() returned None")
            
            # Get the underlying xarray Dataset
            ds_raw = ds.data
            
            logger.info(f"  Loaded: {len(ds_raw.location):,} locations, "
                       f"{len(ds_raw.time):,} timesteps")
            
            # Preprocess
            ds_raw = preprocess_for_merge(ds_raw)
            
            # Rechunk for Zarr
            logger.info("Rechunking for Zarr...")
            chunks = get_optimal_chunks(ds_raw)
            ds_rechunked = ds_raw.chunk(chunks)
            
            # Save to Zarr with proper fill values for integer types
            logger.info(f"Saving to Zarr: {self.output_file}")
            save_to_zarr(ds_rechunked, self.output_file)
            
            logger.info(f"✅ Archive created successfully")
            
        else:
            # Large dataset - process in batches using hierarchical merge
            logger.info(f"Large dataset - will process in batches of {self.batch_size}")
            logger.info("Creating temporary batches in memory, then merging...")
            logger.info("")
            
            # Create temporary workspace for batches
            workspace = Path('merge_workspace_pro')
            workspace.mkdir(exist_ok=True)
            
            # Split files into batches
            batches = create_file_batches(all_files, batch_size=self.batch_size)
            
            logger.info(f"Created {len(batches)} batches")
            logger.info("")
            
            # Process each batch and save as temporary Zarr
            batch_files = []
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)}: {len(batch)} files")
                
                # Read this batch - create temporary file list
                # xsnow.read() can handle a list of paths
                try:
                    ds = xsnow.read(
                        [str(f) for f in batch],  # Pass list of file paths as strings
                        lazy=False,
                        logger=logger
                    )
                except Exception as e:
                    logger.error(f"  Failed to read batch {i+1}: {e}")
                    logger.warning(f"  Skipping this batch")
                    continue
                
                if ds is None or ds.data is None:
                    logger.warning(f"  Batch {i+1} returned no data, skipping")
                    continue
                
                ds_raw = ds.data
                logger.info(f"  Loaded: {len(ds_raw.location):,} locations")
                
                # Preprocess
                ds_raw = preprocess_for_merge(ds_raw)
                
                # Rechunk
                chunks = get_optimal_chunks(ds_raw)
                ds_rechunked = ds_raw.chunk(chunks)
                
                # Save to temporary Zarr with proper fill values
                batch_file = workspace / f'batch_{i:04d}.zarr'
                logger.info(f"  Saving to {batch_file.name}")
                save_to_zarr(ds_rechunked, batch_file)
                batch_files.append(batch_file)
                
                # Clean up
                del ds, ds_raw, ds_rechunked
                gc.collect()
                
                logger.info(f"  ✅ Batch {i+1} complete")
                logger.info("")
            
            # Now merge all the batch Zarr files using hierarchical merge
            logger.info(f"All batches created. Starting hierarchical merge of {len(batch_files)} Zarr archives...")
            logger.info("")
            
            # Use the existing _combine_batches method
            self._combine_batches(workspace)
            
            # Clean up workspace
            logger.info("Cleaning up temporary workspace...")
            shutil.rmtree(workspace)
            logger.info("✅ Workspace cleaned")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ DIRECT PROCESSING COMPLETE")
        logger.info("=" * 70)
        """
        Preprocess datasets before merging.
        
        Handles both old-format (1D slope coords) and new-format (2D data vars).
        """
        # Convert old-format slope coordinates to data variables
        for coord in ['azimuth', 'aspect', 'inclination']:
            if coord in ds.coords and ds.coords[coord].dims == ('slope',):
                ds = ds.reset_coords(coord)
        
        # Drop non-essential coordinates to avoid conflicts
        essential = [
            'location', 'time', 'layer', 'slope', 'realization',
            'lat', 'lon', 'latitude', 'longitude',
            'x', 'y', 'z', 'altitude', 'elevation'
        ]
        
        coords_to_drop = [c for c in ds.coords if c not in essential]
        if coords_to_drop:
            ds = ds.drop_vars(coords_to_drop, errors='ignore')
        
        return ds
    
    def _combine_batches(self, batch_dir, merge_method='hierarchical'):
        """
        Combine batch files into single archive using Zarr.
        
        Uses hierarchical merging to handle large numbers of files efficiently.
        """
        batch_files = sorted(Path(batch_dir).glob('batch_*.zarr'))
        
        if not batch_files:
            raise ValueError(f"No batch files found in {batch_dir}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("HIERARCHICAL MERGE - ZARR VERSION")
        logger.info("=" * 70)
        logger.info(f"Combining {len(batch_files)} batches into single Zarr archive...")
        
        # Create merge workspace
        workspace = Path('merge_workspace')
        workspace.mkdir(exist_ok=True)
        logger.info(f"Merge workspace: {workspace}/ (batch files stay safe!)")
        logger.info("")
        
        # Calculate statistics
        total_size = sum(f.stat().st_size for f in batch_files)
        sizes = [f.stat().st_size for f in batch_files]
        logger.info("Batch file statistics:")
        logger.info(f"  Total size: {total_size / (1024**3):.2f} GB")
        logger.info(f"  Smallest: {min(sizes) / (1024**2):.0f} MB")
        logger.info(f"  Largest: {max(sizes) / (1024**2):.0f} MB")
        logger.info(f"  Average: {(total_size / len(batch_files)) / (1024**2):.0f} MB")
        logger.info("")
        
        # Hierarchical merge
        round_num = 1
        current_files = batch_files
        
        while len(current_files) > 1:
            logger.info(f"ROUND {round_num}: Merging {len(current_files)} files")
            
            next_round_files = []
            group_size = 2  # Pairs
            
            for i in range(0, len(current_files), group_size):
                group = current_files[i:i + group_size]
                
                if len(group) == 1:
                    # Odd file out, carry forward
                    next_round_files.append(group[0])
                    continue
                
                # Merge pair
                pair_num = (i // group_size) + 1
                total_pairs = (len(current_files) + 1) // 2
                
                logger.info(f"  Pair {pair_num}/{total_pairs}: {group[0].name} + {group[1].name}")
                
                out_file = workspace / f'round{round_num}_{i:04d}.zarr'
                
                try:
                    # Detect file types - check each file individually
                    # Some pairs might mix .nc and .zarr if odd files get carried forward
                    datasets = []
                    for f in group:
                        if f.suffix == '.zarr' or f.is_dir():
                            # Zarr directory
                            datasets.append(xr.open_zarr(str(f)))
                        else:
                            # NetCDF file
                            ds_temp = xr.open_dataset(str(f), engine='h5netcdf')
                            ds_temp = preprocess_for_merge(ds_temp)
                            datasets.append(ds_temp)
                    
                    # Concatenate all datasets
                    if len(datasets) == 1:
                        ds = datasets[0]
                    else:
                        # Use coords='minimal' to handle dimension mismatches
                        ds = xr.concat(datasets, dim='location', join='outer',
                                      coords='minimal', data_vars='all')
                    
                    logger.info(f"    Rechunking for Zarr...")
                    # Rechunk to uniform sizes (required for Zarr)
                    chunks = get_optimal_chunks(ds)
                    ds_rechunked = ds.chunk(chunks)
                    
                    logger.info(f"    Saving to Zarr...")
                    # Clean up if exists
                    if out_file.exists():
                        shutil.rmtree(out_file)
                    
                    # Save with proper fill values
                    # Zarr v3 applies default compression automatically
                    save_to_zarr(ds_rechunked, out_file)
                    
                    ds.close()
                    del ds, ds_rechunked
                    gc.collect()
                    
                    # Get size
                    import subprocess
                    result = subprocess.run(
                        ['du', '-sh', str(out_file)],
                        capture_output=True,
                        text=True
                    )
                    size_str = result.stdout.split()[0] if result.returncode == 0 else "?"
                    logger.info(f"    ✅ {size_str}")
                    
                    next_round_files.append(out_file)
                    
                except Exception as e:
                    logger.error(f"    ❌ Failed: {e}")
                    raise
            
            current_files = next_round_files
            round_num += 1
            logger.info("")
        
        # Final file
        final_file = current_files[0]
        
        logger.info("=" * 70)
        logger.info("FINALIZING")
        logger.info("=" * 70)
        logger.info(f"Moving {final_file.name} to {self.output_file}")
        
        # Clean up output if exists
        if self.output_file.exists():
            logger.info(f"Removing existing {self.output_file}")
            shutil.rmtree(self.output_file)
        
        # Move final to output location
        shutil.move(str(final_file), str(self.output_file))
        
        # Clean up workspace
        logger.info("Cleaning up workspace...")
        for f in workspace.glob('*.zarr'):
            if f.exists():
                shutil.rmtree(f)
        workspace.rmdir()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ MERGE COMPLETE")
        logger.info("=" * 70)
        
        # Get final size
        result = subprocess.run(['du', '-sh', str(self.output_file)],
                               capture_output=True, text=True)
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            logger.info(f"Final archive: {self.output_file}")
            logger.info(f"Size: {size_str}")
        
        # Verify
        logger.info("")
        logger.info("Verifying archive...")
        ds_final = xr.open_zarr(self.output_file)
        logger.info(f"  Locations: {len(ds_final.location):,}")
        logger.info(f"  Time steps: {len(ds_final.time):,}")
        logger.info(f"  Variables: {len(ds_final.data_vars)}")
        logger.info(f"  azimuth format: {'data_var ✅' if 'azimuth' in ds_final.data_vars else 'coordinate ⚠️'}")
        ds_final.close()
        
        return self.output_file


def detect_input_type(input_path: Path) -> tuple[str, Path]:
    """
    Auto-detect input type and return (mode, source_path).
    
    Returns:
        ('batches', path): Directory contains batch files → use batch merge
        ('direct', path): Directory contains .pro/.smet files → use direct processing
    
    Raises:
        ValueError: No valid input files found
    """
    logger.info("Auto-detecting input type...")
    
    # Check for batch files
    # Pattern 1: temp_archive subdirectory (common convention)
    temp_archive = input_path / 'temp_archive'
    if temp_archive.exists():
        nc_files = list(temp_archive.glob('*.nc'))
        zarr_files = list(temp_archive.glob('*.zarr'))
        if nc_files or zarr_files:
            logger.info(f"  ✅ Found {len(nc_files)} .nc + {len(zarr_files)} .zarr files in temp_archive/")
            return ('batches', temp_archive)
    
    # Pattern 2: Batch files directly in input directory
    nc_files = list(input_path.glob('batch_*.nc'))
    zarr_files = list(input_path.glob('batch_*.zarr'))
    if nc_files or zarr_files:
        logger.info(f"  ✅ Found {len(nc_files)} .nc + {len(zarr_files)} .zarr batch files")
        return ('batches', input_path)
    
    # Pattern 3: Any .nc or .zarr files (might be batches)
    nc_files = list(input_path.glob('*.nc'))
    zarr_files = list(input_path.glob('*.zarr'))
    if nc_files or zarr_files:
        logger.info(f"  ✅ Found {len(nc_files)} .nc + {len(zarr_files)} .zarr files")
        logger.info(f"  Assuming these are batch files")
        return ('batches', input_path)
    
    # Check for .pro or .smet files
    snowpack_files = find_snowpack_files(input_path, pattern='both')
    if snowpack_files:
        pro_count = sum(1 for f in snowpack_files if f.suffix == '.pro')
        smet_count = sum(1 for f in snowpack_files if f.suffix == '.smet')
        logger.info(f"  ✅ Found {pro_count} .pro + {smet_count} .smet files")
        return ('direct', input_path)
    
    # Nothing found
    raise ValueError(f"No valid input files found in {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate SNOWPACK archives into Zarr format'
    )
    
    parser.add_argument(
        'input_dir',
        nargs='?',
        type=Path,
        help='Directory containing .pro/.smet files (not used with --from-batches)'
    )
    
    parser.add_argument(
        '--from-batches',
        type=Path,
        help='Directory containing existing batch files to merge'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output Zarr archive path (e.g., archive.zarr)'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=365,
        help='Time window in days (default: 365)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2000,
        help='Files per batch (default: 2000)'
    )
    
    parser.add_argument(
        '--compression-level',
        type=int,
        default=5,
        choices=range(0, 10),
        help='Blosc compression level 0-9 (default: 5, 0=none)'
    )
    
    parser.add_argument(
        '--create-transfer-copy',
        action='store_true',
        help='Create transfer copy after merge'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing batches'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel merging (not recommended for Zarr)'
    )
    
    args = parser.parse_args()
    
    # Validate - now more flexible
    if not args.from_batches and not args.input_dir:
        parser.error("Either provide input_dir or --from-batches")
    
    # Determine input mode
    if args.from_batches:
        # Explicit batch mode
        mode = 'batches'
        source = args.from_batches
        logger.info("Mode: BATCH MERGE (forced with --from-batches)")
    elif args.input_dir:
        # Auto-detect
        try:
            mode, source = detect_input_type(args.input_dir)
        except ValueError as e:
            logger.error(str(e))
            return 1
    else:
        parser.error("No input specified")
    
    logger.info("")
    
    # Print configuration
    logger.info("=" * 70)
    logger.info("SNOWPACK ARCHIVE CONSOLIDATION - ZARR VERSION")
    logger.info("=" * 70)
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Source: {source}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Window: {args.window} days")
    
    if mode == 'direct':
        logger.info(f"Batch size: {args.batch_size:,} files per iteration")
    
    logger.info(f"Compression: {'ON (Zarr v3 default)' if args.compression_level > 0 else 'OFF'}")
    logger.info(f"Transfer copy: {'YES' if args.create_transfer_copy else 'NO'}")
    logger.info(f"Resume mode: {'YES' if args.resume else 'NO'}")
    
    if mode == 'batches':
        logger.info(f"Parallel merge: {'YES (not recommended)' if args.parallel else 'NO'}")
    
    logger.info("")
    
    # Create archiver
    archiver = SNOWPACKArchiverZarr(
        output_file=args.output,
        window_days=args.window,
        batch_size=args.batch_size,
        compression_level=args.compression_level,
        create_transfer_copy=args.create_transfer_copy,
        resume=args.resume,
        parallel_merge=args.parallel
    )
    
    # Validate source exists
    if not source.exists():
        logger.error(f"Source directory not found: {source}")
        return 1
    
    # Execute based on mode
    if mode == 'batches':
        logger.info("Strategy: BATCH MERGE")
        logger.info("  • Low memory usage (<5 GB)")
        logger.info("  • Fast hierarchical merging")
        logger.info("  • Batch files will NOT be deleted")
        logger.info("")
        
        archiver._combine_batches(source)
        
    elif mode == 'direct':
        logger.info("Strategy: DIRECT PROCESSING")
        logger.info("  • Processes .pro/.smet files directly")
        logger.info("  • Medium memory usage (10-20 GB)")
        logger.info("  • Creates temporary batches if dataset is large")
        logger.info("")
        
        archiver._process_pro_files(source)
        
    else:
        logger.error(f"Unknown mode: {mode}")
        return 1
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ ALL OPERATIONS COMPLETE")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())