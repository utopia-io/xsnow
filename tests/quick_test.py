from pathlib import Path
from xsnow import read

ROOT = Path(__file__).parent
DATA_PATH = ROOT / 'data' / 'pros' / 'gridded'
path = DATA_PATH / 'VIR1A.pro'
print(DATA_PATH)
print(path)
print(path.exists())
ds = read(path)

print(ds.data_vars)  # What variables exist?
print(list(ds.coords))  # What coordinates exist?