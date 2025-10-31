#!/usr/bin/env python3
"""
Quick setup script for performance testing infrastructure.

Usage:
    python setup_performance_testing.py
    
This will:
1. Check dependencies
2. Create test directory structure
3. Run example benchmarks
4. Generate sample reports
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    print("="*70)
    
    required = {
        'pytest': 'pytest',
        'pytest-benchmark': 'pytest_benchmark',
        'psutil': 'psutil',
    }
    
    optional = {
        'memory_profiler': 'memory_profiler',
    }
    
    missing_required = []
    missing_optional = []
    
    for name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"✅ {name:20s} - installed")
        except ImportError:
            print(f"❌ {name:20s} - missing (REQUIRED)")
            missing_required.append(name)
    
    for name, import_name in optional.items():
        try:
            __import__(import_name)
            print(f"✅ {name:20s} - installed")
        except ImportError:
            print(f"⚠️  {name:20s} - missing (optional)")
            missing_optional.append(name)
    
    print("="*70)
    
    if missing_required:
        print("\n❌ Missing required packages!")
        print("Install with:")
        print(f"    pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print("\n⚠️  Optional packages missing (memory profiling features disabled)")
        print("Install with:")
        print(f"    pip install {' '.join(missing_optional)}")
    
    print("\n✅ All required dependencies installed!")
    return True


def setup_directory_structure():
    """Create performance test directory structure."""
    print("\nSetting up directory structure...")
    print("="*70)
    
    dirs = [
        'tests/performance',
        '.benchmarks',
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
        else:
            print(f"⚠️  Exists:  {dir_path}")
    
    print("="*70)


def run_example_benchmark():
    """Run a quick example benchmark."""
    print("\nRunning example benchmark...")
    print("="*70)
    
    # Create simple example benchmark
    example_code = '''
import pytest
import time

def slow_function():
    """Example slow function."""
    time.sleep(0.01)
    return sum(range(10000))

def fast_function():
    """Example fast function."""
    return sum(range(10000))

class TestExample:
    def test_benchmark_slow(self, benchmark):
        result = benchmark(slow_function)
        assert result > 0
    
    def test_benchmark_fast(self, benchmark):
        result = benchmark(fast_function)
        assert result > 0
'''
    
    # Write example
    example_file = Path('tests/performance/test_example.py')
    if not example_file.exists():
        example_file.write_text(example_code)
        print(f"✅ Created example: {example_file}")
    
    # Run it
    print("\nRunning benchmark...")
    try:
        result = subprocess.run(
            ['pytest', str(example_file), '--benchmark-only', '-v'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ Benchmark completed successfully!")
        else:
            print("⚠️  Benchmark had issues")
            print(result.stderr)
    
    except subprocess.TimeoutExpired:
        print("⚠️  Benchmark timed out")
    except FileNotFoundError:
        print("❌ pytest not found in PATH")
    
    print("="*70)


def generate_report():
    """Generate setup report."""
    print("\n")
    print("="*70)
    print(" PERFORMANCE TESTING SETUP COMPLETE")
    print("="*70)
    
    print("""
Next Steps:
    
1. Review the performance testing guide:
   📄 PERFORMANCE_TESTING_GUIDE.md
   
2. Copy the utility files to your project:
   📁 tests/performance/profiling_utils.py
   📁 tests/performance/test_benchmarks.py
   📁 tests/performance/test_parser_optimization_example.py
   
3. Run your first benchmarks:
   🔧 pytest tests/performance/ --benchmark-only
   
4. Set up CI/CD integration:
   📄 See PERFORMANCE_TESTING_GUIDE.md section on CI/CD
   
Quick Commands:
    
    # Run all benchmarks
    pytest tests/performance/ --benchmark-only
    
    # Save baseline
    pytest tests/performance/ --benchmark-save=baseline
    
    # Compare after changes
    pytest tests/performance/ --benchmark-compare=baseline
    
    # Memory profiling
    pytest tests/performance/ -v --memprof
    
    # Run example parser optimization
    python -m tests.performance.test_parser_optimization_example
    
Files Created:
    ✓ tests/performance/              - Test directory
    ✓ tests/performance/test_example.py   - Example benchmark
    ✓ .benchmarks/                    - Benchmark history
    
Documentation:
    📄 PERFORMANCE_TESTING_GUIDE.md       - Complete guide
    📄 profiling_utils.py                 - Utility functions
    📄 test_benchmarks.py                 - Benchmark suite
    📄 test_parser_optimization_example.py - Real-world example
    📄 pytest_benchmark.ini               - Configuration

Troubleshooting:
    - If benchmarks are noisy: Close other apps, use more runs
    - If memory tests fail: Install memory_profiler
    - For help: See PERFORMANCE_TESTING_GUIDE.md
    """)
    
    print("="*70)
    print("\n✅ Setup complete! Happy benchmarking! 🚀\n")


def main():
    """Main setup function."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     Performance Testing Infrastructure Setup                     ║
║     For xsnow project                                             ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)
    
    # Step 2: Setup directories
    setup_directory_structure()
    
    # Step 3: Run example
    run_example_benchmark()
    
    # Step 4: Generate report
    generate_report()


if __name__ == "__main__":
    main()
