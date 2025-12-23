import xsnow
from xsnow import largescale
from pathlib import Path

def main():
    # Check methods are available
    methods = largescale.list_extensions()
    print(f"Available: {methods}") 

    data_path = Path("./tests/data/pros/gridded")
    print(f"Data path: {data_path}")
    print(f"Data path exists: {data_path.exists()}")
    print(f"Number of files: {len(list(data_path.glob('*')))}")

    # This call triggers parallel scanning/multiprocessing
    ds = xsnow.read(data_path)
    
    if ds is not None:
        print(f"Has method: {hasattr(ds, 'update_from_changed_files')}")
        print(f"Dataset loaded: {ds}")

if __name__ == '__main__':
    main()